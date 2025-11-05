# train_eval_v5_hier_fastcpu.py
# Hierarchical, CPU-accelerated (sklearn-intelex) v5 with:
# - noise scrubbing + char-grams
# - coarse brand boosts
# - fine hard-overrides
# - small LR+ComplementNB ensemble per fine bucket
# - per-class calibrated thresholds
# - review routing for low-confidence
import os, sys, time, warnings, re, json
import numpy as np
import pandas as pd

# 1) optional Intel/oneDAL acceleration
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Extension for Scikit-learn* enabled (https://github.com/uxlfoundation/scikit-learn-intelex)")
except Exception:
    pass

from sklearn.metrics import classification_report, f1_score, confusion_matrix
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Dict, Tuple

from featurizer_patch import normalize_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from lexicon_patch import COARSE_KEYS, coarse_boost, fine_hard_override

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42)

DATA_DIR = "data"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ----------------------------
# Label spaces and hierarchy
# ----------------------------
FINE_LABELS = [
  'DINING','ENTERTAINMENT','FEES','FUEL','GROCERIES','HEALTH','MOBILITY',
  'OTHER','SHOPPING_ECOM','SHOPPING_ELECTRONICS','TRAVEL',
  'UTILITIES_POWER','UTILITIES_TELECOM','UTILITIES_WATER_GAS'
]

COARSE_TO_FINE = {
  'FOOD':              ['DINING','GROCERIES'],
  'HEALTH':            ['HEALTH'],
  'SHOPPING':          ['SHOPPING_ECOM','SHOPPING_ELECTRONICS','ENTERTAINMENT'],
  'TRAVEL':            ['TRAVEL','FUEL','MOBILITY'],
  'UTILITIES':         ['UTILITIES_POWER','UTILITIES_TELECOM','UTILITIES_WATER_GAS'],
  'OTHER_MISC':        ['OTHER','FEES'],
}

FINE_TO_COARSE = {}
for c, kids in COARSE_TO_FINE.items():
    for k in kids:
        FINE_TO_COARSE[k] = c

# defensive: ensure all labels mapped
for lbl in FINE_LABELS:
    if lbl not in FINE_TO_COARSE:
        FINE_TO_COARSE[lbl] = 'OTHER_MISC'

# ----------------------------
# CLI args and featurizer builder
# ----------------------------
def build_text_union_fast(word_max: int, char_max: int, word_min_df: int = 2, char_min_df: int = 2):
    word = TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=word_min_df,
        sublinear_tf=True,
        max_features=word_max,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=char_min_df,
        max_features=char_max,
        dtype=np.float32,
    )
    return FeatureUnion([('w', word), ('c', char)])

def _augment_hierarchy(base_map: Dict[str, List[str]], fines: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str]]:
    """Return (coarse_to_fine, fine_to_coarse, fine_labels) augmented with any labels
    found in the data but not present in the base mapping. Unknowns are mapped to a
    reasonable coarse bucket using simple heuristics.
    """
    ctf = {k: list(v) for k, v in base_map.items()}
    # flatten existing
    existing = set(sum(ctf.values(), []))

    def guess(lbl: str) -> str:
        if lbl in {'DINING','GROCERIES'}:
            return 'FOOD'
        if lbl.startswith('UTILITIES_'):
            return 'UTILITIES'
        if lbl in {'TRAVEL','FUEL','MOBILITY'}:
            return 'TRAVEL'
        if lbl in {'SHOPPING_ECOM','SHOPPING_ELECTRONICS','ENTERTAINMENT','EDUCATION','HOME_IMPROVEMENT','SUBSCRIPTIONS'}:
            return 'SHOPPING'
        if lbl == 'HEALTH':
            return 'HEALTH'
        return 'OTHER_MISC'

    for lbl in fines:
        if lbl not in existing:
            c = guess(lbl)
            ctf.setdefault(c, []).append(lbl)
            existing.add(lbl)

    f2c: Dict[str, str] = {}
    for c, kids in ctf.items():
        for k in kids:
            f2c[k] = c
    finelist = sorted(set(fines))
    return ctf, f2c, finelist

# ----------------------------
# Threshold guards
# ----------------------------
COARSE_MIN = {
  'SHOPPING': 0.50,
  'FOOD':     0.22,
  'TRAVEL':   0.22,
  'UTILITIES':0.22,
  'HEALTH':   0.18,
  'OTHER_MISC':0.16
}
FINE_MIN = {
  'DINING': 0.30, 'GROCERIES': 0.30,
  'SHOPPING_ECOM': 0.25, 'SHOPPING_ELECTRONICS': 0.25, 'ENTERTAINMENT': 0.20,
  'TRAVEL': 0.28, 'FUEL': 0.22, 'MOBILITY': 0.22,
  'UTILITIES_POWER': 0.22, 'UTILITIES_TELECOM': 0.22, 'UTILITIES_WATER_GAS': 0.22,
  'HEALTH': 0.20, 'OTHER': 0.16, 'FEES': 0.20
}

LOW_CONF_ROUTE = 0.60

def _get_feature_names(vec) -> List[str]:
    try:
        return list(vec.get_feature_names_out())
    except Exception:
        # Attempt manual union
        names = []
        try:
            w = vec.transformer_list[0][1]
            c = vec.transformer_list[1][1]
            names = list(getattr(w, 'get_feature_names_out', lambda: [])()) + list(getattr(c, 'get_feature_names_out', lambda: [])())
        except Exception:
            pass
        return names

def _top_k_lr_features(xrow, lr_model: LogisticRegression, class_index: int, feature_names: List[str], k: int = 6) -> List[Tuple[str, float]]:
    try:
        coef = lr_model.coef_[class_index]
        # sparse non-zero indices
        idx = xrow.indices
        data = xrow.data
        contribs = []
        for j, val in zip(idx, data):
            if j < len(coef):
                score = float(coef[j] * val)
                fname = feature_names[j] if j < len(feature_names) else f'f{j}'
                contribs.append((fname, score))
        contribs.sort(key=lambda t: t[1], reverse=True)
        return contribs[:k]
    except Exception:
        return []

def _pick_columns(df: pd.DataFrame):
    if {'raw_text','label'}.issubset(df.columns):
        return 'raw_text','label'
    if {'narrative','category_id'}.issubset(df.columns):
        return 'narrative','category_id'
    if {'narrative','category_display_name'}.issubset(df.columns):
        return 'narrative','category_display_name'
    raise AssertionError('Could not find suitable text/label columns')


def read_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path  = os.path.join(DATA_DIR, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    tcol, lcol = _pick_columns(train_df)
    tcol2, lcol2 = _pick_columns(test_df)
    train_df = train_df.rename(columns={tcol:'raw_text', lcol:'label'})
    test_df  = test_df.rename(columns={tcol2:'raw_text', lcol2:'label'})

    # Optional weak supervision
    weak_path = os.path.join(DATA_DIR, "weak_train.csv")
    if os.path.exists(weak_path):
        weak = pd.read_csv(weak_path)
        if {'raw_text','label'}.issubset(set(weak.columns)):
            weak['__weight__'] = 0.30
            train_df['__weight__'] = 1.0
            train_df = pd.concat([train_df[['raw_text','label','__weight__']], weak[['raw_text','label','__weight__']]], ignore_index=True)
        else:
            train_df['__weight__'] = 1.0
    else:
        train_df['__weight__'] = 1.0

    return train_df, test_df

def show_counts(df, name):
    cnt = df['label'].value_counts().sort_index()
    print(f"{name} counts:\n", cnt)

def build_fine_classifier(n_jobs: int = -1):
    # small, fast ensemble
    lr = LogisticRegression(max_iter=2000, n_jobs=n_jobs, solver='saga', C=2.0, class_weight='balanced')
    nb = ComplementNB(alpha=0.5)  # great for imbalanced token text
    clf = VotingClassifier(
        estimators=[('lr', lr), ('nb', nb)],
        voting='soft', weights=[0.7, 0.3], n_jobs=n_jobs
    )
    return clf

def compute_class_thresholds(y_true, proba, classes, min_floor=None):
    # per-class F1-optimal thresholds on a calibration slice
    thresholds = {}
    for i, cls in enumerate(classes):
        y = (y_true == cls).astype(int)
        p = proba[:, i]
        if len(np.unique(y)) < 2:
            thresholds[cls] = 0.50
            continue
        # try 101 evenly spaced thresholds (fast, smooth)
        tgrid = np.linspace(0.05, 0.95, 101)
        best_t, best_f1 = 0.50, 0.0
        for t in tgrid:
            yhat = (p >= t).astype(int)
            if yhat.sum() == 0:
                continue
            f1 = f1_score(y, yhat)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        if min_floor and cls in min_floor:
            best_t = max(best_t, min_floor[cls])
        thresholds[cls] = float(best_t)
    return thresholds

def main():
    ap = argparse.ArgumentParser(description='Fast hierarchical categorizer (CPU) with progress bars')
    ap.add_argument('--word-max-features', type=int, default=200_000)
    ap.add_argument('--char-max-features', type=int, default=150_000)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--coarse-cal-size', type=int, default=12_000)
    ap.add_argument('--fine-cal-frac', type=float, default=0.2)
    ap.add_argument('--low-conf', type=float, default=0.60)
    ap.add_argument('--save-model', action='store_true')
    ap.add_argument('--word-min-df', type=int, default=2)
    ap.add_argument('--char-min-df', type=int, default=2)
    args = ap.parse_args()

    global LOW_CONF_ROUTE
    LOW_CONF_ROUTE = args.low_conf

    start_all = time.time()
    train_df, test_df = read_data()
    show_counts(train_df, "Train")
    show_counts(test_df, "Test")

    # Shared vectorizer
    print("\n[1/6] Fit shared featurizer …")
    vec = build_text_union_fast(args.word_max_features, args.char_max_features, args.word_min_df, args.char_min_df)
    X_train = vec.fit_transform(train_df['raw_text'])
    X_test  = vec.transform(test_df['raw_text'])

    # Encode coarse and fine labels
    y_fine_train = train_df['label'].values
    y_fine_test  = test_df['label'].values

    # Dynamically augment hierarchy to include all labels present
    all_fines = sorted(list(set(y_fine_train) | set(y_fine_test)))
    global COARSE_TO_FINE, FINE_TO_COARSE, FINE_LABELS
    COARSE_TO_FINE, FINE_TO_COARSE, FINE_LABELS = _augment_hierarchy(COARSE_TO_FINE, all_fines)

    # derive y_coarse
    y_coarse_train = np.array([FINE_TO_COARSE.get(lbl, 'OTHER_MISC') for lbl in y_fine_train])
    y_coarse_test  = np.array([FINE_TO_COARSE.get(lbl, 'OTHER_MISC') for lbl in y_fine_test])

    # ---------------- COARSE LR -----------------
    print("\n[2/6] Fit COARSE LR …")
    le_coarse = LabelEncoder().fit(COARSE_KEYS)
    y_coarse_idx = le_coarse.transform(y_coarse_train)
    coarse_lr = LogisticRegression(max_iter=3000, solver='saga', n_jobs=args.n_jobs, C=2.0, multi_class='multinomial', class_weight='balanced')
    coarse_lr.fit(X_train, y_coarse_idx, sample_weight=train_df['__weight__'].values)

    # Calibrate coarse (prob thresholds per class)
    print("[3/6] Calibrate COARSE …\n")
    # small holdout for coarse thresholds
    cal_idx = np.random.RandomState(42).choice(len(train_df), size=min(args.coarse_cal_size, len(train_df)//5), replace=False)
    X_cal, y_cal = X_train[cal_idx], y_coarse_train[cal_idx]
    coarse_proba_cal = coarse_lr.predict_proba(X_cal)
    coarse_thresholds = compute_class_thresholds(
        y_true=y_cal, proba=coarse_proba_cal, classes=COARSE_KEYS, min_floor=COARSE_MIN
    )
    print("Per-class thresholds:")
    for k in COARSE_KEYS:
        print(f"  {k:<24} -> {coarse_thresholds[k]:.2f}")

    # ---------------- FINE models (one per coarse) -----------------
    print("\n[4/6] Fit FINE models (parallel) …")
    fine_models = {}
    fine_encoders = {}
    fine_thresholds = {}
    fine_solo = {}  # NEW: buckets with exactly one fine label

    for coarse_key in tqdm(COARSE_KEYS, desc='Train fine buckets'):
        kids = COARSE_TO_FINE.get(coarse_key, [])
        mask = np.isin(y_fine_train, kids)
        if mask.sum() == 0:
            continue

        Xk = X_train[mask]
        yk = y_fine_train[mask]
        wk = train_df.loc[mask, '__weight__'].values

        # NEW: handle single-label buckets (e.g., HEALTH)
        uniq = np.unique(yk)
        if uniq.size < 2:
            sole = uniq[0]
            fine_solo[coarse_key] = sole
            fine_models[coarse_key] = None
            fine_encoders[coarse_key] = None
            fine_thresholds[coarse_key] = {sole: FINE_MIN.get(sole, 0.20)}
            print(f"[fine:{coarse_key}] single-label bucket → always '{sole}'")
            continue

        # normal multi-label path
        le = LabelEncoder().fit(kids)
        yk_idx = le.transform(yk)

        clf = build_fine_classifier(args.n_jobs)
        clf.fit(Xk, yk_idx, sample_weight=wk)
        fine_models[coarse_key] = clf
        fine_encoders[coarse_key] = le

        # thresholds per coarse group
        ncal = min(2000, max(200, int(args.fine_cal_frac * Xk.shape[0])))
        cal_sel = np.random.RandomState(123).choice(Xk.shape[0], size=ncal, replace=False)
        prob_cal = clf.predict_proba(Xk[cal_sel])
        y_cal_k = yk[cal_sel]
        classes_k = list(le.classes_)
        thr_k = compute_class_thresholds(y_true=y_cal_k, proba=prob_cal, classes=classes_k, min_floor=FINE_MIN)
        fine_thresholds[coarse_key] = thr_k


    # ---------------- Predict hierarchy -----------------
    print("\n[5/6] Predict hierarchy (batch) …")
    raw_texts = test_df['raw_text'].tolist()
    y_true = test_df['label'].values

    # coarse probabilities (with brand bumps)
    coarse_proba = coarse_lr.predict_proba(X_test)
    # apply boosts and thresholds for routing
    routed_coarse = []
    coarse_idx_to_key = {i:k for i,k in enumerate(le_coarse.classes_)}
    for i, row_p in enumerate(coarse_proba):
        boosted = coarse_boost(row_p.copy(), raw_texts[i])
        # respect threshold floors: if top prob below its threshold, fall back to OTHER_MISC
        top_idx = int(np.argmax(boosted))
        top_key = coarse_idx_to_key[top_idx]
        thr = coarse_thresholds.get(top_key, 0.2)
        if boosted[top_idx] < thr:
            top_key = 'OTHER_MISC'
        routed_coarse.append(top_key)
    routed_coarse = np.array(routed_coarse)

    # fine predictions within routed bucket
    y_pred = []
    y_prob = []
    y_expl = []
    feat_names = _get_feature_names(vec)
    for i in tqdm(range(len(test_df)), desc='Predict fine rows'):
        ck = routed_coarse[i]
        txt = raw_texts[i]

        # NEW: single-label bucket → deterministic prediction
        if ck in fine_solo:
            lbl = fine_solo[ck]
            pmax = 0.99
            lbl = fine_hard_override(lbl, txt)  # still allow brand corrections (mostly no-op here)
            y_pred.append(lbl)
            y_prob.append(pmax)
            y_expl.append("")
            continue

        if ck not in fine_models:
            y_pred.append('OTHER')
            y_prob.append(0.0)
            y_expl.append("")
            continue

        model = fine_models[ck]
        enc = fine_encoders[ck]
        proba = model.predict_proba(X_test[i])
        proba = np.asarray(proba)[0]
        classes_k = list(enc.classes_)

        j = int(np.argmax(proba))
        lbl = classes_k[j]
        pmax = float(proba[j])

        thr_map = fine_thresholds.get(ck, {})
        thr = thr_map.get(lbl, FINE_MIN.get(lbl, 0.2))
        if pmax < thr and 'OTHER' in classes_k:
            lbl = 'OTHER'
            pmax = float(proba[classes_k.index('OTHER')])

        lbl = fine_hard_override(lbl, txt)
        y_pred.append(lbl)
        y_prob.append(pmax)
        # explanation (low-confidence only): use LR contributions when available
        try:
            nem = getattr(model, 'named_estimators_', {})
            lr_part = nem.get('lr') if isinstance(nem, dict) else None
            if lr_part is not None and pmax < LOW_CONF_ROUTE:
                xrow = X_test[i]
                top = _top_k_lr_features(xrow, lr_part, j, feat_names, k=6)
                y_expl.append('; '.join([f"{a}:{b:.2f}" for a,b in top]))
            else:
                y_expl.append("")
        except Exception:
            y_expl.append("")


    # ---------------- Reporting -----------------
    print("\n[6/6] Final report …\n")
    print("=== Hierarchical Hybrid (fastcpu) ===")
    macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
    print("Macro-F1:", macro_f1)
    print(classification_report(y_true, y_pred, digits=3, labels=FINE_LABELS))
    print("Labels order:", FINE_LABELS)
    cm = confusion_matrix(y_true, y_pred, labels=FINE_LABELS)
    print(cm)
    # Persist reports
    pd.DataFrame(cm, index=FINE_LABELS, columns=FINE_LABELS).to_csv(os.path.join(REPORT_DIR, 'confusion_matrix.csv'))
    with open(os.path.join(REPORT_DIR, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump({
            'macro_f1': float(macro_f1),
            'labels': FINE_LABELS,
            'n_train': int(len(train_df)),
            'n_test': int(len(test_df)),
            'coarse_keys': list(COARSE_KEYS),
            'coarse_to_fine': COARSE_TO_FINE,
        }, fh, indent=2)

    # route low-confidence to review.csv
    review_mask = np.array(y_prob) < LOW_CONF_ROUTE
    routed = int(review_mask.sum())
    if routed > 0:
        out = test_df.copy()
        out['pred'] = y_pred
        out['pred_proba'] = y_prob
        out['explain'] = y_expl
        out_low = out[review_mask][['raw_text','label','pred','pred_proba','explain']]
        os.makedirs(DATA_DIR, exist_ok=True)
        out_low.to_csv(os.path.join(DATA_DIR, "review.csv"), index=False)
    # save preds
    out_all = test_df.copy()
    out_all['pred'] = y_pred
    out_all['pred_proba'] = y_prob
    out_all['explain'] = y_expl
    out_path = os.path.join(REPORT_DIR, "preds_test.csv")
    out_all.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    if routed:
        print(f"Routed {routed} low-confidence rows (<{LOW_CONF_ROUTE}) to data/review.csv")

    # Optional: persist model bundle for inference
    if 'args' in locals() and getattr(args, 'save_model', False):
        try:
            from joblib import dump
            mdl_dir = Path('models'); mdl_dir.mkdir(parents=True, exist_ok=True)
            dump({
                'vectorizer': vec,
                'coarse_lr': coarse_lr,
                'fine_models': fine_models,
                'fine_encoders': fine_encoders,
                'fine_thresholds': fine_thresholds,
                'coarse_thresholds': coarse_thresholds,
                'le_coarse': le_coarse,
                'coarse_to_fine': COARSE_TO_FINE,
                'fine_to_coarse': FINE_TO_COARSE,
                'fine_labels': FINE_LABELS,
            }, mdl_dir / 'txcat_fastcpu.joblib')
            print("Saved models/txcat_fastcpu.joblib")
        except Exception as e:
            print("Model save skipped:", e)

if __name__ == "__main__":
    main()
