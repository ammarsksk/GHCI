# src/train_eval_v2.py
import os, re, time, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from rules_v2 import load_taxonomy, rules_predict

CONF_THRESH = 0.60  # <--- confidence router threshold

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

AMT_BINS = [(0,100),(100,500),(500,1500),(1500,5000),(5000,20000),(20000,1e12)]
def amount_bucket(a: str) -> str:
    try:
        val = float(re.sub(r"[^\d.]", "", a))
    except:
        return "amt_unknown"
    for lo, hi in AMT_BINS:
        if lo <= val < hi:
            return f"amt_{int(lo)}_{int(hi if hi<1e6 else 999999)}"
    return "amt_unknown"

PHRASE_FLAGS = [
    ("has_refund", r"\brefund\b"),
    ("has_cashback", r"\bcashback\b"),
    ("has_reversal", r"\breversal|chargeback\b"),
    ("has_emi", r"\bemi\b"),
    ("has_bill", r"\bbill( desk)?|bill payment|recharge|water bill|gas bill\b"),
    ("has_ticket", r"\bticket\b"),
    ("has_fuel", r"\bfuel|petrol|diesel|surcharge\b"),
]

def augment_text(df):
    # add special tokens to help word model
    toks = []
    toks.append("__chan_"+df.get("channel","").lower())
    toks.append("__bank_"+df.get("bank","").lower())
    toks.append("__dir_"+df.get("direction","").lower())
    toks.append("__amt_"+amount_bucket(str(df.get("amount",""))))
    text = df["text_norm"]
    for name, rgx in PHRASE_FLAGS:
        if re.search(rgx, df["text_norm"]):
            toks.append("__"+name)
    return text + " " + " ".join(toks)

def load_csv(path):
    df = pd.read_csv(path)
    df["text_norm"] = df["raw_text"].astype(str).apply(normalize)
    df["text_aug"]  = df.apply(augment_text, axis=1)
    df["amount_bucket"] = df["amount"].astype(str).apply(amount_bucket)
    for k in ["channel","bank","direction"]:
        if k in df.columns: df[k] = df[k].astype(str).str.lower().fillna("unk")
    return df

def build_pipeline():
    # OneHotEncoder arg differs across sklearn; handle both
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
            ("meta", ohe, ["channel","bank","direction","amount_bucket"]),
        ],
        remainder="drop", sparse_threshold=1.0, n_jobs=-1
    )
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=3.0, max_iter=2500,
        n_jobs=-1, verbose=1, class_weight="balanced"
    )
    pipe = Pipeline([("col", col), ("clf", clf)])
    return pipe

def split_calib(train_df, frac=0.10, seed=42):
    # stratified holdout for calibration
    cal = train_df.groupby("label", group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=seed))
    core = train_df.drop(index=cal.index).reset_index(drop=True)
    cal = cal.reset_index(drop=True)
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

def batched_predict_proba(estimator, X_df, batch_size=15000):
    probs=[]
    for i in tqdm(range(0, len(X_df), batch_size), desc="Predict proba (ML)", unit="rows"):
        chunk = X_df.iloc[i:i+batch_size]
        probs.append(estimator.predict_proba(chunk))
    return np.vstack(probs)

def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    train = load_csv("data/train.csv")
    test  = load_csv("data/test.csv") if os.path.exists("data/test.csv") else train.sample(frac=0.3, random_state=42)

    print("Train counts:\n", train["label"].value_counts())
    print("Test counts:\n",  test["label"].value_counts())

    # split calibration holdout
    train_core, calib = split_calib(train, frac=0.10)
    print(f"Calibration holdout: {len(calib)} rows")

    Xtr = train_core[["text_norm","text_aug","channel","bank","direction","amount_bucket"]]
    ytr = train_core["label"]
    Xcal = calib[["text_norm","text_aug","channel","bank","direction","amount_bucket"]]
    ycal = calib["label"]
    Xte = test[["text_norm","text_aug","channel","bank","direction","amount_bucket"]]
    yte = test["label"]

    pipe = build_pipeline()
    t0=time.time(); print("\n[1/4] Fit core model …")
    pipe.fit(Xtr, ytr)
    print(f"Fit done in {time.time()-t0:.1f}s")

    # probability calibration on holdout (Platt/sigmoid)
    print("\n[2/4] Calibrate probabilities (sigmoid) …")
    cal = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")
    cal.fit(Xcal, ycal)

    # ML-only calibrated predictions
    print("\n[3/4] Predict calibrated ML …")
    ml_proba = batched_predict_proba(cal, Xte, batch_size=15000)
    classes = cal.classes_
    ml_idx   = ml_proba.argmax(axis=1)
    ml_label = [classes[i] for i in ml_idx]
    ml_conf  = ml_proba.max(axis=1)

    rep_ml, cm_ml, labs_ml = report("ML-only (calibrated)", yte, ml_label)

    # Hybrid (rules -> calibrated ML), confidence router
    print("\n[4/4] Hybrid rules→ML + confidence routing …")
    tax = load_taxonomy()
    hybrid_label=[]; hybrid_conf=[]; hybrid_src=[]
    review_rows=[]

    for (raw, mllab, mlc) in tqdm(zip(test["raw_text"], ml_label, ml_conf), total=len(ml_label), unit="rows", desc="Hybrid pass"):
        r = rules_predict(raw, tax)
        if r and r[1] >= 0.94:
            lab, conf, det = r
            hybrid_label.append(lab); hybrid_conf.append(conf); hybrid_src.append("rules")
        else:
            hybrid_label.append(mllab); hybrid_conf.append(float(mlc)); hybrid_src.append("ml")

        if hybrid_conf[-1] < CONF_THRESH:
            review_rows.append({"raw_text": raw, "pred_label": hybrid_label[-1], "pred_conf": hybrid_conf[-1]})

    rep_h, cm_h, labs_h = report("Hybrid (rules→ML calibrated)", yte, hybrid_label)

    # write artifacts
    joblib.dump(cal, "models/txcat_pipeline_calibrated.joblib")
    with open("models/labels.json","w") as f: json.dump(list(classes), f, indent=2)
    with open("models/metrics.json","w") as f:
        json.dump({
            "ml_only_calibrated":{"macro_f1":rep_ml["macro avg"]["f1-score"], "cm":cm_ml, "labels":labs_ml},
            "hybrid_calibrated": {"macro_f1":rep_h["macro avg"]["f1-score"],  "cm":cm_h,  "labels":labs_h}
        }, f, indent=2)

    # confidence routing output
    if review_rows:
        pd.DataFrame(review_rows).to_csv("data/review.csv", index=False, encoding="utf-8")
        print(f"\nRouted {len(review_rows)} low-confidence rows (<{CONF_THRESH}) to data/review.csv")
    else:
        print("\nNo rows fell below confidence threshold.")

    # store scored predictions for error analysis
    out = pd.DataFrame({
        "id": test["id"], "raw_text": test["raw_text"], "true_label": yte,
        "ml_label": ml_label, "ml_conf": ml_conf,
        "hybrid_label": hybrid_label, "hybrid_conf": hybrid_conf, "source": hybrid_src
    })
    out.to_csv("reports/preds_test.csv", index=False, encoding="utf-8")
    print("Saved reports/preds_test.csv")

if __name__=="__main__":
    main()
