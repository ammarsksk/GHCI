# make_weak_from_review.py
# Mine high-precision weak labels from data/review.csv to harden decision boundaries.
import os
import pandas as pd
from lexicon_patch import fine_hard_override

def main():
    in_path = "data/review.csv"
    out_path = "data/weak_train.csv"
    if not os.path.exists(in_path):
        print("review.csv not found at data/review.csv")
        return
    df = pd.read_csv(in_path)
    if 'raw_text' not in df.columns:
        # try common column names
        cand = [c for c in df.columns if 'text' in c.lower()]
        if cand: df = df.rename(columns={cand[0]:'raw_text'})
    df['weak_label'] = df['raw_text'].apply(lambda s: fine_hard_override('UNKNOWN', s))
    weak = df[df['weak_label']!='UNKNOWN'][['raw_text','weak_label']].copy()
    weak = weak.rename(columns={'weak_label':'label'})
    weak.to_csv(out_path, index=False)
    print(f"Wrote {len(weak)} weak rows -> {out_path}")

if __name__ == "__main__":
    main()
