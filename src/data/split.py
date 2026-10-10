import os
from sklearn.model_selection import GroupShuffleSplit
from src.data.load import load_data

def make_splits(df, seed=42):
    # split by question so one question never appears in two splits
    # 80% train, then the other 20% halved into val and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, temp_idx = next(gss.split(df, groups=df["input"]))
    train, temp = df.iloc[tr_idx], df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    va_idx, te_idx = next(gss2.split(temp, groups=temp["input"]))
    return train, temp.iloc[va_idx], temp.iloc[te_idx]

if __name__ == "__main__":
    df = load_data()
    # check: should show 3000 rows per LLM
    print(df["label"].value_counts())
    os.makedirs("data/processed", exist_ok=True)
    # save each split as a CSV in data/processed, with its LLM counts
    for name, part in zip(["train", "val", "test"], make_splits(df)):
        part.to_csv(f"data/processed/{name}.csv", index=False)
        print(name, len(part), part["label"].value_counts().to_dict())