import os
from sklearn.model_selection import train_test_split
from src.data.load import load_data

def make_splits(df, seed=42):
    # 80% train, then split the other 20% in half for val and test
    # stratify keeps the 6 LLMs evenly balanced in every split
    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    return train, val, test

if __name__ == "__main__":
    df = load_data()
    # check: should show 3000 rows per LLM
    print(df["label"].value_counts())
    os.makedirs("data/processed", exist_ok=True)
    # save each split as a CSV in data/processed
    for name, part in zip(["train", "val", "test"], make_splits(df)):
        part.to_csv(f"data/processed/{name}.csv", index=False)
        print(name, len(part))