# train_eval_v6_faststack.py
# Faster hierarchical categorizer with batched prediction, progress bars, and
# configurable parallelism. Uses word+char TF‑IDF (float32), a coarse LR
# router with brand boosts, and per‑coarse fine ensembles (LR+CNB).

import os, time, warnings, json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Optional Intel/oneDAL acceleration
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("Extension for Scikit-learn* enabled (sklearn-intelex)")
except Exception:
    pass

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from tqdm.auto import tqdm
import argparse
from scipy.sparse import vstack, csr_matrix
from joblib import Parallel, delayed

from featurizer_patch import normalize_text
from lexicon_patch import COARSE_KEYS, coarse_boost, fine_hard_override

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = "data"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ----------------------------
# Featurizer
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


def transform_in_chunks(vec, texts: List[str], chunk_size: int, desc: str) -> csr_matrix:
    mats = []
    for i in tqdm(range(0, len(texts), chunk_size), desc=desc):
        chunk = texts[i:i+chunk_size]
        mats.append(vec.transform(chunk))
    return vstack(mats, format='csr') if len(mats) > 1 else (mats[0] if mats else csr_matrix((0,0)))


def predict_proba_in_chunks(model, X, chunk_size: int, desc: str) -> np.ndarray:
    parts = []
    for i in tqdm(range(0, X.shape[0], chunk_size), desc=desc):
        parts.append(model.predict_proba(X[i:i+chunk_size]))
    return np.vstack(parts)


# ----------------------------
# Hierarchy helpers
# ----------------------------
BASE_COARSE_TO_FINE = {
  'FOOD':              ['DINING','GROCERIES'],
  'HEALTH':            ['HEALTH'],
  'SHOPPING':          ['SHOPPING_ECOM','SHOPPING_ELECTRONICS','ENTERTAINMENT'],
  'TRAVEL':            ['TRAVEL','FUEL','MOBILITY'],
  'UTILITIES':         ['UTILITIES_POWER','UTILITIES_TELECOM','UTILITIES_WATER_GAS'],
  'OTHER_MISC':        ['OTHER','FEES'],
}

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


def _augment_hierarchy(base_map: Dict[str, List[str]], fines: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str]]:
    ctf = {k: list(v) for k, v in base_map.items()}
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
            ctf.setdefault(guess(lbl), []).append(lbl)
            existing.add(lbl)

    f2c: Dict[str, str] = {}
    for c, kids in ctf.items():
        for k in kids:
            f2c[k] = c
    finelist = sorted(set(fines))
    return ctf, f2c, finelist


def _pick_columns(df: pd.DataFrame) -> Tuple[str, str]:
    if {'raw_text','label'}.issubset(df.columns):
        return 'raw_text','label'
    if {'narrative','category_id'}.issubset(df.columns):
        return 'narrative','category_id'
    if {'narrative','category_display_name'}.issubset(df.columns):
        return 'narrative','category_display_name'
    raise AssertionError('Could not find suitable text/label columns')


def compute_class_thresholds(y_true, proba, classes, min_floor=None):
    thresholds = {}
    for i, cls in enumerate(classes):
        y = (y_true == cls).astype(int)
        p = proba[:, i]
        if len(np.unique(y)) < 2:
            thresholds[cls] = 0.50
            continue
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


def compute_class_thresholds_parallel(y_true, proba, classes, min_floor=None, n_jobs: int = 1):
    def _one(i_cls):
        i, cls = i_cls
        y = (y_true == cls).astype(int)
        p = proba[:, i]
        if len(np.unique(y)) < 2:
            thr = 0.50
        else:
            tgrid = np.linspace(0.05, 0.95, 101)
            best_t, best_f1 = 0.50, 0.0
            for t in tgrid:
                yhat = (p >= t).astype(int)
                if yhat.sum() == 0:
                    continue
                f1 = f1_score(y, yhat)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thr = best_t
        if min_floor and cls in min_floor:
            thr = max(thr, min_floor[cls])
        return (cls, float(thr))

    items = list(enumerate(classes))
    if n_jobs and n_jobs != 1:
        out = Parallel(n_jobs=n_jobs)(delayed(_one)(it) for it in tqdm(items, desc='Calibrate thresholds'))
    else:
        out = [_one(it) for it in tqdm(items, desc='Calibrate thresholds')]
    return {k: v for k, v in out}


def _get_feature_names(vec) -> List[str]:
    try:
        return list(vec.get_feature_names_out())
    except Exception:
        names = []
        try:
            w = vec.transformer_list[0][1]
            c = vec.transformer_list[1][1]
            names = list(getattr(w, 'get_feature_names_out', lambda: [])()) + list(getattr(c, 'get_feature_names_out', lambda: [])())
        except Exception:
            pass
        return names


def build_fine_classifier(n_jobs: int = -1):
    lr = LogisticRegression(max_iter=2000, n_jobs=n_jobs, solver='saga', C=2.0, class_weight='balanced')
    nb = ComplementNB(alpha=0.5)
    clf = VotingClassifier(
        estimators=[('lr', lr), ('nb', nb)],
        voting='soft', weights=[0.7, 0.3], n_jobs=n_jobs
    )
    return clf


def main():
    ap = argparse.ArgumentParser(description='v6 fast hierarchical categorizer')
    ap.add_argument('--word-max-features', type=int, default=150_000)
    ap.add_argument('--char-max-features', type=int, default=100_000)
    ap.add_argument('--word-min-df', type=int, default=2)
    ap.add_argument('--char-min-df', type=int, default=2)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--coarse-cal-size', type=int, default=12_000)
    ap.add_argument('--fine-cal-frac', type=float, default=0.2)
    ap.add_argument('--low-conf', type=float, default=0.60)
    ap.add_argument('--save-model', action='store_true')
    ap.add_argument('--chunk-size', type=int, default=25000, help='Batch size for vectorizer transform and predict')
    ap.add_argument('--bucket-workers', type=int, default=1, help='Parallel workers for training fine buckets (outer parallelism)')
    ap.add_argument('--predict-workers', type=int, default=0, help='Parallel workers for per-bucket prediction (0=sequential)')
    ap.add_argument('--coarse-trainer', choices=['lr','sgd'], default='lr', help='Coarse router trainer: LogisticRegression or SGD (log loss)')
    ap.add_argument('--coarse-max-iter', type=int, default=2000)
    ap.add_argument('--coarse-tol', type=float, default=1e-3)
    ap.add_argument('--coarse-epochs', type=int, default=2)
    ap.add_argument('--coarse-batch', type=int, default=10000)
    ap.add_argument('--threshold-workers', type=int, default=0, help='Parallel jobs for per-class threshold search (0=sequential)')
    args = ap.parse_args()

    LOW_CONF_ROUTE = args.low_conf

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    tcol, lcol = _pick_columns(train_df)
    tcol2, lcol2 = _pick_columns(test_df)
    train_df = train_df.rename(columns={tcol:'raw_text', lcol:'label'})
    test_df  = test_df.rename(columns={tcol2:'raw_text', lcol2:'label'})
    train_df['__weight__'] = 1.0

    print('Train size:', len(train_df), 'Test size:', len(test_df))

    # Vectorize
    print("[1/6] Fit featurizer")
    vec = build_text_union_fast(args.word_max_features, args.char_max_features, args.word_min_df, args.char_min_df)
    vec.fit(train_df['raw_text'])
    X_train = transform_in_chunks(vec, train_df['raw_text'].tolist(), args.chunk_size, desc='Vectorize train')
    X_test  = transform_in_chunks(vec, test_df['raw_text'].tolist(), args.chunk_size, desc='Vectorize test')

    # Labels and hierarchy
    y_fine_train = train_df['label'].values
    y_fine_test  = test_df['label'].values
    COARSE_TO_FINE, FINE_TO_COARSE, FINE_LABELS = _augment_hierarchy(BASE_COARSE_TO_FINE, sorted(list(set(y_fine_train) | set(y_fine_test))))
    y_coarse_train = np.array([FINE_TO_COARSE.get(lbl, 'OTHER_MISC') for lbl in y_fine_train])
    y_coarse_test  = np.array([FINE_TO_COARSE.get(lbl, 'OTHER_MISC') for lbl in y_fine_test])

    # Coarse model
    print("[2/6] Fit COARSE router")
    le_coarse = LabelEncoder().fit(COARSE_KEYS)
    y_coarse_idx = le_coarse.transform(y_coarse_train)

    def fit_coarse_router():
        if args.coarse_trainer == 'sgd':
            # SGD with log loss; explicit progress via partial_fit
            clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, random_state=42, tol=args.coarse_tol)
            classes = np.arange(len(COARSE_KEYS))
            idx_all = np.arange(X_train.shape[0])
            for ep in range(1, args.coarse_epochs + 1):
                rng = np.random.RandomState(42 + ep)
                rng.shuffle(idx_all)
                for s in tqdm(range(0, len(idx_all), args.coarse_batch), desc=f'Coarse SGD epoch {ep}/{args.coarse_epochs}'):
                    sel = idx_all[s:s+args.coarse_batch]
                    if ep == 1 and s == 0:
                        clf.partial_fit(X_train[sel], y_coarse_idx[sel], classes=classes)
                    else:
                        clf.partial_fit(X_train[sel], y_coarse_idx[sel])
            return clf
        else:
            # LogisticRegression (saga), with verbose output
            lr = LogisticRegression(max_iter=args.coarse_max_iter, solver='saga', n_jobs=args.n_jobs, C=2.0, multi_class='multinomial', class_weight='balanced', tol=args.coarse_tol, verbose=2)
            tqdm.write(f'Fitting LogisticRegression(saga) max_iter={args.coarse_max_iter} tol={args.coarse_tol} n_jobs={args.n_jobs}')
            lr.fit(X_train, y_coarse_idx, sample_weight=train_df['__weight__'].values)
            return lr

    coarse_lr = fit_coarse_router()

    # Calibrate thresholds for coarse
    print("[3/6] Calibrate COARSE thresholds")
    cal_idx = np.random.RandomState(42).choice(len(train_df), size=min(args.coarse_cal_size, len(train_df)//5), replace=False)
    X_cal, y_cal = X_train[cal_idx], y_coarse_train[cal_idx]
    coarse_proba_cal = coarse_lr.predict_proba(X_cal)
    # Use only the coarse classes learned by the model, mapped back to keys
    coarse_classes_idx = getattr(coarse_lr, 'classes_', None)
    if coarse_classes_idx is None:
        coarse_classes_idx = np.unique(y_coarse_idx)
    coarse_classes_keys = list(le_coarse.inverse_transform(coarse_classes_idx))
    if args.threshold_workers and args.threshold_workers > 0:
        coarse_thresholds = compute_class_thresholds_parallel(y_true=y_cal, proba=coarse_proba_cal, classes=coarse_classes_keys, min_floor=COARSE_MIN, n_jobs=args.threshold_workers)
    else:
        coarse_thresholds = compute_class_thresholds(y_true=y_cal, proba=coarse_proba_cal, classes=coarse_classes_keys, min_floor=COARSE_MIN)

    # Fine models per coarse bucket
    print("[4/6] Fit FINE buckets")
    fine_models: Dict[str, VotingClassifier] = {}
    fine_encoders: Dict[str, LabelEncoder] = {}
    fine_thresholds: Dict[str, Dict[str, float]] = {}
    fine_solo: Dict[str, str] = {}

    # Heuristic to avoid oversubscription: if outer parallel >1, set inner n_jobs=1
    inner_jobs = 1 if args.bucket_workers and args.bucket_workers > 1 else args.n_jobs

    def train_bucket(coarse_key: str):
        kids = COARSE_TO_FINE.get(coarse_key, [])
        mask = np.isin(y_fine_train, kids)
        if mask.sum() == 0:
            return (coarse_key, None, None, {}, None)
        Xk = X_train[mask]
        yk = y_fine_train[mask]
        uniq = np.unique(yk)
        if uniq.size < 2:
            return (coarse_key, None, None, {uniq[0]: FINE_MIN.get(uniq[0], 0.20)}, uniq[0])
        le = LabelEncoder().fit(kids)
        yk_idx = le.transform(yk)
        clf = build_fine_classifier(inner_jobs)
        clf.fit(Xk, yk_idx)
        ncal = min(2000, max(200, int(args.fine_cal_frac * Xk.shape[0])))
        cal_sel = np.random.RandomState(123).choice(Xk.shape[0], size=ncal, replace=False)
        prob_cal = clf.predict_proba(Xk[cal_sel])
        y_cal_k = yk[cal_sel]
        # Align proba columns to actual model classes and map back to label strings
        model_class_idx = getattr(clf, 'classes_', None)
        if model_class_idx is None:
            model_class_idx = np.unique(yk_idx)
        classes_k = list(le.inverse_transform(model_class_idx))
        thr_k = compute_class_thresholds(y_true=y_cal_k, proba=prob_cal, classes=classes_k, min_floor=FINE_MIN)
        return (coarse_key, clf, le, thr_k, None)

    keys = list(COARSE_KEYS)
    if args.bucket_workers and args.bucket_workers > 1:
        results = Parallel(n_jobs=args.bucket_workers, prefer='threads')(delayed(train_bucket)(k) for k in tqdm(keys, desc='Dispatch fine buckets'))
    else:
        results = [train_bucket(k) for k in tqdm(keys, desc='Train fine buckets')]

    for ck, clf, le, thr, solo in results:
        if solo is not None:
            fine_solo[ck] = solo
            fine_thresholds[ck] = thr
        elif clf is not None:
            fine_models[ck] = clf
            fine_encoders[ck] = le
            fine_thresholds[ck] = thr

    # Predict
    print("[5/6] Predict hierarchy")
    raw_texts = test_df['raw_text'].tolist()
    coarse_proba = predict_proba_in_chunks(coarse_lr, X_test, args.chunk_size, desc='Coarse routing')
    routed_coarse = []
    # Map probability columns to actual coarse keys learned by the model
    coarse_classes_idx = getattr(coarse_lr, 'classes_', None)
    if coarse_classes_idx is None:
        coarse_classes_idx = np.arange(coarse_proba.shape[1])
    coarse_classes_keys = list(le_coarse.inverse_transform(coarse_classes_idx))
    coarse_idx_to_key = {i: k for i, k in enumerate(coarse_classes_keys)}
    for i, row_p in enumerate(coarse_proba):
        boosted = coarse_boost(row_p.copy(), raw_texts[i])
        top_idx = int(np.argmax(boosted))
        top_key = coarse_idx_to_key[top_idx]
        thr = coarse_thresholds.get(top_key, 0.2)
        routed_coarse.append(top_key if boosted[top_idx] >= thr else 'OTHER_MISC')
    routed_coarse = np.array(routed_coarse)

    # Fine predictions in batches per bucket
    y_true = test_df['label'].values
    y_pred: List[str] = []
    y_prob: List[float] = []
    y_expl: List[str] = []
    feat_names = _get_feature_names(vec)
    bucket_to_indices: Dict[str, List[int]] = {}
    for i, ck in enumerate(routed_coarse):
        bucket_to_indices.setdefault(ck, []).append(i)

    # Optionally parallelize per-bucket predictions
    def predict_bucket(ck: str, idxs: List[int]):
        outs = []
        if ck in fine_solo:
            lbl = fine_solo[ck]
            for i in idxs:
                outs.append((i, lbl, 0.99, ""))
            return outs
        if ck not in fine_models or fine_models[ck] is None:
            for i in idxs:
                outs.append((i, 'OTHER', 0.0, ""))
            return outs
        model = fine_models[ck]
        enc = fine_encoders[ck]
        probas = model.predict_proba(X_test[idxs])
        # Map model columns (encoded indices) back to label strings for this bucket
        model_class_idx = getattr(model, 'classes_', None)
        if model_class_idx is None:
            model_class_idx = np.arange(probas.shape[1])
        labels_for_cols = list(enc.inverse_transform(model_class_idx))
        for row_i, proba in zip(idxs, probas):
            proba = np.asarray(proba)
            j = int(np.argmax(proba))
            lbl = labels_for_cols[j]
            pmax = float(proba[j])
            thr_map = fine_thresholds.get(ck, {})
            thr = thr_map.get(lbl, FINE_MIN.get(lbl, 0.2))
            if pmax < thr and 'OTHER' in labels_for_cols:
                lbl = 'OTHER'
                pmax = float(proba[labels_for_cols.index('OTHER')])
            lbl = fine_hard_override(lbl, raw_texts[row_i])
            # skip heavy explanation by default; keep for low conf only
            expl = ""
            try:
                nem = getattr(model, 'named_estimators_', {})
                lr_part = nem.get('lr') if isinstance(nem, dict) else None
                if lr_part is not None and pmax < LOW_CONF_ROUTE:
                    xrow = X_test[row_i]
                    coef = lr_part.coef_[j]
                    idx = xrow.indices
                    data = xrow.data
                    pairs = []
                    for jj, val in zip(idx, data):
                        if jj < len(coef):
                            fname = feat_names[jj] if jj < len(feat_names) else f'f{jj}'
                            pairs.append((fname, float(coef[jj] * val)))
                    pairs.sort(key=lambda t: t[1], reverse=True)
                    expl = '; '.join([f"{a}:{b:.2f}" for a,b in pairs[:6]])
            except Exception:
                pass
            outs.append((row_i, lbl, pmax, expl))
        return outs

    items = list(bucket_to_indices.items())
    if args.predict_workers and args.predict_workers > 0:
        par_out = Parallel(n_jobs=args.predict_workers, prefer='threads')(delayed(predict_bucket)(ck, idxs) for ck, idxs in tqdm(items, desc='Dispatch predict buckets'))
        flat = [t for part in par_out for t in part]
    else:
        flat = []
        for ck, idxs in tqdm(items, desc='Predict fine buckets'):
            flat.extend(predict_bucket(ck, idxs))

    # assemble in original order
    y_pred = [None] * len(test_df)
    y_prob = [0.0] * len(test_df)
    y_expl = [""] * len(test_df)
    for i, lbl, pmax, expl in flat:
        y_pred[i] = lbl
        y_prob[i] = pmax
        y_expl[i] = expl

    # Report
    print("[6/6] Report")
    macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 3)
    print("Macro-F1:", macro_f1)
    print(classification_report(y_true, y_pred, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true) | set(y_pred))))
    pd.DataFrame(cm).to_csv(os.path.join(REPORT_DIR, 'confusion_matrix.csv'), index=False)
    with open(os.path.join(REPORT_DIR, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump({
            'macro_f1': float(macro_f1),
            'n_train': int(len(train_df)),
            'n_test': int(len(test_df)),
        }, fh, indent=2)

    # Feedback routing
    LOW_CONF_ROUTE_VAL = LOW_CONF_ROUTE
    review_mask = np.array(y_prob) < LOW_CONF_ROUTE_VAL
    if review_mask.any():
        out = test_df.copy()
        out['pred'] = y_pred
        out['pred_proba'] = y_prob
        out['explain'] = y_expl
        out[review_mask][['raw_text','label','pred','pred_proba','explain']].to_csv(os.path.join(DATA_DIR, 'review.csv'), index=False)

    # Save preds
    out_all = test_df.copy()
    out_all['pred'] = y_pred
    out_all['pred_proba'] = y_prob
    out_all['explain'] = y_expl
    out_all.to_csv(os.path.join(REPORT_DIR, 'preds_test.csv'), index=False)

    # Optional save model
    if args.save_model:
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
                'fine_labels': FINE_LABELS,
            }, mdl_dir / 'txcat_fastcpu_v6.joblib')
            print('Saved models/txcat_fastcpu_v6.joblib')
        except Exception as e:
            print('Model save skipped:', e)


if __name__ == '__main__':
    main()
