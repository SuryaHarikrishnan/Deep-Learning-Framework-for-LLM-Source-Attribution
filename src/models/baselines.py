import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.data.dataset import make_text

def main():
    tr = pd.read_csv("data/processed/train.csv")
    te = pd.read_csv("data/processed/test.csv")
    rows = []
    # run the same baseline for the 3 RQ2 settings
    for mode in ["input", "output", "both"]:
        # turn text into word and word-pair importance scores
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        Xtr = vec.fit_transform(make_text(tr, mode))  # learn from train only
        Xte = vec.transform(make_text(te, mode))
        # train a simple classifier and predict on test
        pred = LogisticRegression(max_iter=1000).fit(Xtr, tr["label"]).predict(Xte)
        rows.append({
            "mode": mode,
            "accuracy": accuracy_score(te["label"], pred),
            "macro_f1": f1_score(te["label"], pred, average="macro"),
        })
    out = pd.DataFrame(rows)
    print(out)
    # save the results table for the report
    out.to_csv("results/tables/baseline_tfidf.csv", index=False)

if __name__ == "__main__":
    main()