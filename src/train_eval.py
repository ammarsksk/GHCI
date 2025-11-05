import pandas as pd, numpy as np, joblib, json, re, os, time
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from rules import load_taxonomy, rules_predict

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

def load_csv(path):
    df = pd.read_csv(path)
    df["text_norm"] = df["raw_text"].astype(str).apply(normalize)
    return df

def build_pipeline():
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        min_df=5,                 # slightly higher for big data
        max_features=300_000,     # cap to keep RAM reasonable
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32
    )
    clf = LogisticRegression(
        solver="saga", penalty="l2", C=2.0,
        max_iter=2000, n_jobs=-1, verbose=1  # verbose shows solver progress
    )
    return Pipeline([("vec", vec), ("clf", clf)])

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

def batched_predict(pipe, X, batch_size=20000):
    """Predict in chunks so we can show a progress bar & reduce peak RAM."""
    preds = []
    for i in tqdm(range(0, len(X), batch_size), desc="Predict (ML)", unit="rows"):
        chunk = X.iloc[i:i+batch_size]
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
        print("No external test.csv found → used a random split")

    print("Train counts:\n", train["label"].value_counts())
    print("Test counts:\n",  test["label"].value_counts())

    Xtr, ytr = train["text_norm"], train["label"]
    Xte, yte = test["text_norm"],  test["label"]

    pipe = build_pipeline()

    t0 = time.time()
    print("\n[1/3] Fit vectorizer + classifier (this can take a bit)…")
    pipe.fit(Xtr, ytr)
    print(f"Fit done in {time.time()-t0:.1f}s")

    print("\n[2/3] Predict (ML-only) with progress…")
    ypr_ml = batched_predict(pipe, Xte, batch_size=20000)
    rep_ml, cm_ml, labs_ml = report("ML-only", yte, ypr_ml)

    print("\n[3/3] Apply rules → ML (hybrid) with progress…")
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
