# src/train_eval_v3.py
import os, re, time, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix

# uses your expanded rules
from rules_v2 import load_taxonomy, rules_predict

# ------------------ knobs ------------------
CONF_THRESH = 0.60                 # route low-confidence predictions to data/review.csv
RULE_CONF_STRONG = 0.94            # min conf to even consider a rule
RULE_CONF_NEUTRAL = 0.96           # stricter for neutral classes
BATCH_PROBA = 15000                # batch size for predict_proba

# strong vs weak rule classes (tuned for your data)
STRONG = {"HEALTH","TRAVEL","SHOPPING_ELECTRONICS","ENTERTAINMENT","MOBILITY","GROCERIES"}
WEAK   = {"SHOPPING_ECOM","UTILITIES_POWER","FUEL"}
# ------------------------------------------------


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

AMT_BINS = [(0,100),(100,500),(500,1500),(1500,5000),(5000,20000),(20000,1e12)]
def amount_bucket(a: str) -> str:
    try:
        val = float(re.sub(r"[^\d.]", "", str(a)))
    except:
        return "amt_unknown"
    for lo, hi in AMT_BINS:
        if lo <= val < hi:
            return f"amt_{int(lo)}_{int(hi if hi<1e6 else 999999)}"
    return "amt_unknown"

def extract_mcc(text: str) -> str:
    m = re.search(r"\bmcc\s*(\d{3,4})\b", str(text).lower())
    return f"mcc_{m.group(1)}" if m else "mcc_none"

PHRASE_FLAGS = [
    ("has_refund", r"\brefund\b"),
    ("has_cashback", r"\bcashback\b"),
    ("has_reversal", r"\breversal|chargeback\b"),
    ("has_emi", r"\bemi\b"),
    ("has_bill", r"\bbill( desk)?|bill payment|recharge|water bill|gas bill\b"),
    ("has_ticket", r"\bticket\b"),
    ("has_fuel", r"\bfuel|petrol|diesel|surcharge\b"),
]

def augment_text(row):
    toks = []
    toks.append("__chan_"+str(row.get("channel","")).lower())
    toks.append("__bank_"+str(row.get("bank","")).lower())
    toks.append("__dir_"+str(row.get("direction","")).lower())
    toks.append("__amt_"+amount_bucket(str(row.get("amount",""))))
    text = row["text_norm"]
    for name, rgx in PHRASE_FLAGS:
        if re.search(rgx, row["text_norm"]):
            toks.append("__"+name)
    return text + " " + " ".join(toks)

def load_csv(path):
    df = pd.read_csv(path)
    df["text_norm"] = df["raw_text"].astype(str).apply(normalize)
    df["text_aug"]  = df.apply(augment_text, axis=1)
    df["amount_bucket"] = df["amount"].astype(str).apply(amount_bucket)
    df["mcc_code"] = df["raw_text"].apply(extract_mcc)
    for k in ["channel","bank","direction"]:
        if k in df.columns: df[k] = df[k].astype(str).str.lower().fillna("unk")
    return df

def build_pipeline():
    # OneHotEncoder arg changed in sklearn 1.5
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    col = ColumnTransformer(
        transformers=[
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3,6), min_df=8,
                max_features=400_000, sublinear_tf=True, lowercase=True,
                dtype=np.float32
            ), "text_norm"),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1,2), min_df=5,
                max_features=160_000, sublinear_tf=True, lowercase=True,
                dtype=np.float32
            ), "text_aug"),
            ("meta", ohe, ["channel","bank","direction","amount_bucket","mcc_code"]),
        ],
        remainder="drop", sparse_threshold=1.0, n_jobs=-1
    )
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=3.0, max_iter=2500,
        n_jobs=-1, verbose=1, class_weight="balanced"
    )
    return Pipeline([("col", col), ("clf", clf)])

def split_calib(train_df, frac=0.10, seed=42):
    # stratified holdout without pandas deprecation
    parts = [g.sample(frac=frac, random_state=seed) for _, g in train_df.groupby("label")]
    cal = pd.concat(parts).reset_index(drop=True)
    core = train_df.drop(index=cal.index).reset_index(drop=True)
    return core, cal

def report(title, y_true, y_pred):
    labels = sorted(pd.unique(pd.Series(y_true)))
    rep = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    print(f"\n=== {title} ===")
    print("Macro-F1:", round(rep["macro avg"]["f1-score"], 3))
    print(pd.DataFrame(rep).transpose().round(3))
    print("Labels order:", labels)
    print(np.array(cm))
    return rep, cm, labels

def batched_predict_proba(estimator, X_df, batch_size=BATCH_PROBA):
    probs=[]
    for i in tqdm(range(0, len(X_df), batch_size), desc="Predict proba (ML)", unit="rows"):
        chunk = X_df.iloc[i:i+batch_size]
        probs.append(estimator.predict_proba(chunk))
    return np.vstack(probs)

def tune_thresholds(proba_cal, ycal, classes):
    # pick per-class cutoff to maximize one-vs-rest F1 on calibration holdout
    ycal = np.asarray(ycal)
    thr = {}
    for j, lab in enumerate(classes):
        pj = proba_cal[:, j]
        yj = (ycal == lab).astype(int)
        best_f1, best_t = 0.0, 0.5
        # coarse grid; widen if you like
        for t in [x/100 for x in range(20, 90, 2)]:
            pred = (pj >= t).astype(int)
            tp = np.sum((pred==1) & (yj==1))
            fp = np.sum((pred==1) & (yj==0))
            fn = np.sum((pred==0) & (yj==1))
            f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[lab] = best_t
    print("\nPer-class thresholds (calibration):")
    for k in sorted(thr.keys()):
        print(f"  {k:24s} -> {thr[k]:.2f}")
    return thr

def pick_with_threshold(row_probs, classes, thr_map):
    # choose highest class whose proba >= its threshold; else argmax
    cand = [(p, classes[j]) for j, p in enumerate(row_probs) if p >= thr_map[classes[j]]]
    if cand:
        p, lab = max(cand, key=lambda x: x[0])
        return lab, float(p)
    j = int(np.argmax(row_probs))
    return classes[j], float(row_probs[j])

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    train = load_csv("data/train.csv")
    test  = load_csv("data/test.csv") if os.path.exists("data/test.csv") else train.sample(frac=0.3, random_state=42)

    print("Train counts:\n", train["label"].value_counts())
    print("Test counts:\n",  test["label"].value_counts())

    # split for calibration
    core, calib = split_calib(train, frac=0.10)
    print(f"Calibration holdout: {len(calib)} rows")

    Xtr = core[["text_norm","text_aug","channel","bank","direction","amount_bucket","mcc_code"]]
    ytr = core["label"]
    Xcal = calib[["text_norm","text_aug","channel","bank","direction","amount_bucket","mcc_code"]]
    ycal = calib["label"]
    Xte = test[["text_norm","text_aug","channel","bank","direction","amount_bucket","mcc_code"]]
    yte = test["label"]

    # build + fit
    pipe = build_pipeline()
    t0=time.time(); print("\n[1/5] Fit core model …")
    pipe.fit(Xtr, ytr)
    print(f"Fit done in {time.time()-t0:.1f}s")

    # calibrate (Platt/sigmoid)
    print("\n[2/5] Calibrate probabilities (sigmoid) …")
    cal = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")
    cal.fit(Xcal, ycal)

    # thresholds from calibration
    print("\n[3/5] Compute thresholds on calibration …")
    proba_cal = batched_predict_proba(cal, Xcal, batch_size=BATCH_PROBA)
    classes = cal.classes_
    thr_map  = tune_thresholds(proba_cal, ycal.values, classes)
    idx_of   = {c:i for i,c in enumerate(classes)}

    # ML-only predictions (argmax and thresholded)
    print("\n[4/5] Predict calibrated ML on test …")
    proba_te = batched_predict_proba(cal, Xte, batch_size=BATCH_PROBA)

    ml_argmax_idx = np.argmax(proba_te, axis=1)
    ml_argmax_lab = [classes[j] for j in ml_argmax_idx]
    ml_argmax_conf = proba_te[np.arange(len(proba_te)), ml_argmax_idx]

    ml_thr_lab=[]; ml_thr_conf=[]
    for i in range(len(proba_te)):
        lab, conf = pick_with_threshold(proba_te[i], classes, thr_map)
        ml_thr_lab.append(lab); ml_thr_conf.append(conf)

    rep_ml_arg, cm_ml_arg, labs_ml_arg = report("ML-only (calibrated argmax)", yte, ml_argmax_lab)
    rep_ml_thr, cm_ml_thr, labs_ml_thr = report("ML-only (calibrated + thresholds)", yte, ml_thr_lab)

    # Hybrid (rules -> ML-thresholded) with smart override + routing
    print("\n[5/5] Hybrid rules→ML (thresholded) + smart override + routing …")
    tax = load_taxonomy()
    hybrid_label=[]; hybrid_conf=[]; hybrid_src=[]
    review_rows=[]

    raw_series = test["raw_text"].tolist()
    true_labels = yte.tolist()

    for i, raw in tqdm(list(enumerate(raw_series)), total=len(raw_series), unit="rows", desc="Hybrid pass"):
        r = rules_predict(raw, tax)
        base_lab, base_conf = ml_thr_lab[i], float(ml_thr_conf[i])
        max_prob = float(ml_argmax_conf[i])

        if r and r[1] >= RULE_CONF_STRONG:
            rlab, rconf, _ = r
            use_rule = False
            if rlab in STRONG:
                use_rule = True
            elif rlab in WEAK:
                p_rule = float(proba_te[i, idx_of.get(rlab, 0)])
                # need some ML support or near-top score
                if (p_rule >= 0.35) or (p_rule >= max_prob - 0.08):
                    use_rule = True
            else:
                use_rule = (rconf >= RULE_CONF_NEUTRAL)

            if use_rule:
                pred_lab, pred_conf, src = rlab, float(rconf), "rules"
            else:
                pred_lab, pred_conf, src = base_lab, base_conf, "ml"
        else:
            pred_lab, pred_conf, src = base_lab, base_conf, "ml"

        hybrid_label.append(pred_lab); hybrid_conf.append(pred_conf); hybrid_src.append(src)
        if pred_conf < CONF_THRESH:
            review_rows.append({"raw_text": raw, "pred_label": pred_lab, "pred_conf": pred_conf})

    rep_h, cm_h, labs_h = report("Hybrid (rules→ML thresholds, smart override)", true_labels, hybrid_label)

    # artifacts
    Path("models").mkdir(exist_ok=True)
    joblib.dump(cal, "models/txcat_pipeline_calibrated.joblib")
    with open("models/labels.json","w") as f: json.dump(list(classes), f, indent=2)
    with open("models/thresholds.json","w") as f: json.dump(thr_map, f, indent=2)
    with open("models/metrics.json","w") as f:
        json.dump({
            "ml_only_argmax":       {"macro_f1":rep_ml_arg["macro avg"]["f1-score"], "cm":cm_ml_arg, "labels":labs_ml_arg},
            "ml_only_thresholded":  {"macro_f1":rep_ml_thr["macro avg"]["f1-score"], "cm":cm_ml_thr, "labels":labs_ml_thr},
            "hybrid_smart":         {"macro_f1":rep_h["macro avg"]["f1-score"],     "cm":cm_h,      "labels":labs_h}
        }, f, indent=2)

    # outputs
    out = pd.DataFrame({
        "id": test["id"],
        "raw_text": raw_series,
        "true_label": true_labels,
        "ml_arg_label": ml_argmax_lab,
        "ml_arg_conf":  ml_argmax_conf,
        "ml_thr_label": ml_thr_lab,
        "ml_thr_conf":  ml_thr_conf,
        "hybrid_label": hybrid_label,
        "hybrid_conf":  hybrid_conf,
        "source":       hybrid_src
    })
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/preds_test.csv", index=False, encoding="utf-8")
    print("Saved reports/preds_test.csv")

    if review_rows:
        pd.DataFrame(review_rows).to_csv("data/review.csv", index=False, encoding="utf-8")
        print(f"Routed {len(review_rows)} low-confidence rows (<{CONF_THRESH}) to data/review.csv")
    else:
        print("No rows routed to review.")

if __name__=="__main__":
    main()
