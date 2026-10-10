import re
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# a token is a word, a punctuation mark, or a newline
TOKEN_RE = re.compile(r"\w+|[^\w\s]|\n")
PAD, UNK = 0, 1  # ids for padding and unknown words

def tokenize(text):
    # split text into a list of tokens
    return TOKEN_RE.findall(str(text))

def build_vocab(texts, max_size=30000, min_freq=2):
    # count every token, then give an id to each common one
    counts = Counter(t for x in texts for t in tokenize(x))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for tok, c in counts.most_common(max_size):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab

def make_text(df, mode):
    # choose what the model sees: question, answer, or both
    if mode == "input":
        return df["input"].astype(str).tolist()
    if mode == "output":
        return df["output"].astype(str).tolist()
    return (df["input"].astype(str) + " ||| " + df["output"].astype(str)).tolist()

class LLMDataset(Dataset):
    def __init__(self, df, vocab, label2id, mode="output", max_len=256):
        # labels as numbers
        self.y = torch.tensor(df["label"].map(label2id).tolist())
        rows = []
        for text in make_text(df, mode):
            # words to ids, cut to max_len
            ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
            # pad short texts so every row has the same length
            rows.append(ids + [PAD] * (max_len - len(ids)))
        self.x = torch.tensor(rows)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def get_loaders(mode="output", max_len=256, batch_size=64):
    # read the saved splits
    tr, va, te = [pd.read_csv(f"data/processed/{n}.csv") for n in ["train", "val", "test"]]
    # build vocab from train only so test data does not leak in
    vocab = build_vocab(make_text(tr, mode))
    labels = sorted(tr["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}

    def mk(d, shuffle):
        return DataLoader(LLMDataset(d, vocab, label2id, mode, max_len),
                          batch_size=batch_size, shuffle=shuffle)

    # returns train, val, test loaders plus the vocab and label names
    return mk(tr, True), mk(va, False), mk(te, False), vocab, labels