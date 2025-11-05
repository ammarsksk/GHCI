import joblib, json, argparse, numpy as np
from rules import load_taxonomy, rules_predict, normalize

def load_pipeline():
    pipe = joblib.load("models/txcat_pipeline.joblib")
    with open("models/labels.json") as f:
        labels = json.load(f)
    return pipe, labels

def top_features_for_text(pipe, text_norm, topk=5):
    vec = pipe.named_steps["vec"]
    clf = pipe.named_steps["clf"]
    X = vec.transform([text_norm])
    pred_label = pipe.predict([text_norm])[0]
    classes = list(clf.classes_)
    coef = clf.coef_[classes.index(pred_label)]
    nz = X.nonzero()[1]
    feats = [(vec.get_feature_names_out()[j], float(coef[j]*X[0, j])) for j in nz]
    feats.sort(key=lambda x: x[1], reverse=True)
    return feats[:topk]

def predict(text: str):
    tax = load_taxonomy()
    rule = rules_predict(text, tax)
    pipe, _ = load_pipeline()
    text_norm = normalize(text)

    if rule and rule[1] >= 0.93:
        lab, conf, det = rule
        return {"label": lab, "confidence": conf, "source":"rules",
                "explanation": [f"rule:{det.get('rule')} hit:{det.get('hit','')}"]}

    proba = pipe.predict_proba([text_norm])[0]
    idx = int(np.argmax(proba))
    lab = pipe.named_steps["clf"].classes_[idx]
    conf = float(proba[idx])
    feats = [f"{f}:{w:.2f}" for f,w in top_features_for_text(pipe, text_norm)]
    return {"label": lab, "confidence": conf, "source":"ml", "explanation": feats}

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    print(predict(args.text))
