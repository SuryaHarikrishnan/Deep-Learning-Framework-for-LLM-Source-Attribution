import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

OUT = "results/tables/"


def style_features(text):
    # simple counts that describe how a response is written and formatted
    words = text.split()
    n = max(len(words), 1)
    return {
        "chars": len(text),
        "words": len(words),
        "newlines": text.count("\n"),
        "markdown_symbols": len(re.findall(r"[#*`]", text)),
        "bullets": len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s", text)),
        "punctuation": len(re.findall(r"[^\w\s]", text)),
        "digits": len(re.findall(r"\d", text)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "vocab_variety": len(set(w.lower() for w in words)) / n,  # unique words / total words
    }


def strip_formatting(text):
    # remove markdown symbols and line breaks
    return re.sub(r"\s+", " ", re.sub(r"[#*`|>_~]", "", text))


def first_words(text, k=50):
    # keep only the first k words so every response has the same length
    return " ".join(text.split()[:k])


def tfidf_lr(train_text, test_text, y_train, y_test, token_pattern=None):
    # same TF-IDF + logistic regression setup as the baseline
    kwargs = {"max_features": 50000, "ngram_range": (1, 2)}
    if token_pattern:
        kwargs["token_pattern"] = token_pattern
        kwargs["lowercase"] = False
    vec = TfidfVectorizer(**kwargs)
    Xtr = vec.fit_transform(train_text)
    Xte = vec.transform(test_text)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
    pred = clf.predict(Xte)
    return (accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")), vec, clf


def main():
    tr = pd.read_csv("data/processed/train.csv")
    te = pd.read_csv("data/processed/test.csv")
    ytr, yte = tr["label"], te["label"]
    otr, ote = tr["output"].astype(str), te["output"].astype(str)

    # 1) average style features per LLM
    ftr = pd.DataFrame([style_features(t) for t in otr])
    fte = pd.DataFrame([style_features(t) for t in ote])
    table = ftr.assign(label=ytr.values).groupby("label").mean().round(2)
    print(table)
    table.to_csv(OUT + "rq4_style_features.csv")

    # 2) classify using only the style features (no words)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(ftr, ytr)
    pred = rf.predict(fte)
    rows = [{"setting": "Style features only (random forest)",
             "accuracy": accuracy_score(yte, pred),
             "macro_f1": f1_score(yte, pred, average="macro")}]
    # same classifier using only response length (characters and words)
    rf_len = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    pred_len = rf_len.fit(ftr[["chars", "words"]], ytr).predict(fte[["chars", "words"]])
    rows.append({"setting": "Response length only (random forest)",
                 "accuracy": accuracy_score(yte, pred_len),
                 "macro_f1": f1_score(yte, pred_len, average="macro")})
    importance = pd.Series(rf.feature_importances_, index=ftr.columns).sort_values(ascending=False)
    print(importance.round(3))

    # 3) TF-IDF ablations: remove one kind of signal at a time
    words_and_marks = r"\w+|[^\w\s]|\n"  # words, punctuation and newlines
    marks_only = r"[^\w\s]|\n"           # punctuation and newlines only
    settings = [
        ("Words only (baseline)", otr, ote, None),
        ("Words + punctuation + newlines", otr, ote, words_and_marks),
        ("Formatting stripped (no markdown, no newlines)",
         otr.map(strip_formatting), ote.map(strip_formatting), words_and_marks),
        ("Truncated to first 50 words (length removed)",
         otr.map(first_words), ote.map(first_words), None),
        ("Punctuation and newlines only (no words)", otr, ote, marks_only),
    ]
    baseline_vec = baseline_clf = None
    for name, a, b, pattern in settings:
        t0 = time.time()
        (acc, f1), vec, clf = tfidf_lr(a, b, ytr, yte, pattern)
        rows.append({"setting": name, "accuracy": acc, "macro_f1": f1})
        print(f"{name}: acc {acc:.4f}, f1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if baseline_vec is None:
            baseline_vec, baseline_clf = vec, clf
    out = pd.DataFrame(rows)
    print(out)
    out.to_csv(OUT + "rq4_ablation.csv", index=False)

    # 4) most distinctive words per LLM from the baseline model
    vocab = np.array(baseline_vec.get_feature_names_out())
    top = {}
    for i, label in enumerate(baseline_clf.classes_):
        top[label] = ", ".join(vocab[np.argsort(baseline_clf.coef_[i])[-8:][::-1]])
    top_df = pd.Series(top, name="top_words")
    print(top_df)
    top_df.to_csv(OUT + "rq4_top_words.csv")


if __name__ == "__main__":
    main()
