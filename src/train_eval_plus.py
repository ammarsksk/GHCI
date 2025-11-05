import os, re, time, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from rules import load_taxonomy, rules_predict

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

# --- Feature engineering helpers ---
AMT_BINS = [(0,100),(100,500),(500,1500),(1500,5000),(5000,1e12)]
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
    ("has_bill", r"\bbill( desk)?|bill payment|recharge\b"),
    ("has_ticket", r"\bticket\b"),
    ("has_fuel", r"\bfuel|petrol|diesel|surcharge\b"),
]

def augment_text(row):
    # add special tokens to help the word vectorizer
    toks = []
    toks.append("__chan_"+str(row.get("channel","")).lower())
    toks.append("__bank_"+str(row.get("bank","")).lower())
    toks.append("__dir_"+str(row.get("direction","")).lower())
    toks.append("__amt_"+amount_bucket(str(row.get("amount",""))))
    txt = row["text_norm"]
    for name, rgx in PHRASE_FLAGS:
        if re.search(rgx, row["text_norm"]):
            toks.append("__"+name)
    return txt + " " + " ".join(toks)

def load_csv(path):
    df = pd.read_csv(path)
    df["text_norm"] = df["raw_text"].astype(str).apply(normalize)
    df["text_aug"]  = df.apply(augment_text, axis=1)
    df["amount_bucket"] = df["amount"].astype(str).apply(amount_bucket)
    # clean meta
    for k in ["channel","bank","direction"]:
        if k in df.columns: df[k] = df[k].astype(str).str.lower().fillna("unk")
    return df

from sklearn.preprocessing import OneHotEncoder

def build_pipeline():
    # robust to sklearn versions (1.2+ uses sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # fallback for very old versions
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    col = ColumnTransformer(
        transformers=[
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3,6), min_df=5,
                max_features=350_000, sublinear_tf=True, lowercase=True,
                dtype=np.float32
            ), "text_norm"),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1,2), min_df=3,
                max_features=120_000, sublinear_tf=True, lowercase=True,
                dtype=np.float32
            ), "text_aug"),
            ("meta", ohe, ["channel","bank","direction","amount_bucket"])
        ],
        remainder="drop", sparse_threshold=1.0, n_jobs=-1
    )

    clf = LogisticRegression(
        solver="saga", penalty="l2", C=2.5, max_iter=2500,
        n_jobs=-1, verbose=1, class_weight="balanced"
    )
    return Pipeline([("col", col), ("clf", clf)])


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

def batched_predict(pipe, X_df, batch_size=20000):
    preds=[]
    for i in tqdm(range(0, len(X_df), batch_size), desc="Predict (ML)", unit="rows"):
        chunk = X_df.iloc[i:i+batch_size]
        preds.extend(pipe.predict(chunk))
    return preds

def main():
    Path("models").mkdir(exist_ok=True)
    train = load_csv("data/train.csv")
    if os.path.exists("data/test.csv"):
        test = load_csv("data/test.csv")
        print("Using external test.csv")
    else:
        test = train.sample(frac=0.2, random_state=42)
        train = train.drop(index=test.index).reset_index(drop=True)
        test  = test.reset_index(drop=True)
        print("No external test.csv found → random split")

    print("Train counts:\n", train["label"].value_counts())
    print("Test counts:\n",  test["label"].value_counts())

    Xtr, ytr = train[["text_norm","text_aug","channel","bank","direction","amount_bucket"]], train["label"]
    Xte, yte = test[ ["text_norm","text_aug","channel","bank","direction","amount_bucket"]],  test["label"]

    pipe = build_pipeline()

    t0 = time.time()
    print("\n[1/3] Fit multi-view pipeline…")
    pipe.fit(Xtr, ytr)
    print(f"Fit done in {time.time()-t0:.1f}s")

    print("\n[2/3] Predict (ML-only) with progress…")
    ypr_ml = batched_predict(pipe, Xte, batch_size=20000)
    rep_ml, cm_ml, labs_ml = report("ML-only", yte, ypr_ml)

    print("\n[3/3] Hybrid (rules→ML) with progress…")
    tax = load_taxonomy()
    ypr_h = []
    for raw_text, mlp in tqdm(zip(test["raw_text"], ypr_ml), total=len(ypr_ml), desc="Hybrid rules", unit="rows"):
        rule = rules_predict(raw_text, tax)
        ypr_h.append(rule[0] if (rule and rule[1] >= 0.93) else mlp)
    rep_h, cm_h, labs_h = report("Hybrid (rules→ML)", yte, ypr_h)

    joblib.dump(pipe, "models/txcat_pipeline.joblib")
    with open("models/labels.json","w") as f: json.dump(sorted(train["label"].unique()), f)
    with open("models/metrics.json","w") as f:
        json.dump({
            "ml_only":  {"macro_f1":rep_ml["macro avg"]["f1-score"], "cm":cm_ml, "labels":labs_ml},
            "hybrid":   {"macro_f1":rep_h["macro avg"]["f1-score"],  "cm":cm_h,  "labels":labs_h}
        }, f, indent=2)

if __name__=="__main__":
    main()
