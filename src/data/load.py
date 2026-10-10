import pandas as pd

# the 6 LLM families we classify
LLMS = ["ChatGPT", "Claude", "Gemini", "Llama", "Qwen", "DeepSeek"]
RAW_PATH = "data/raw/TXD-22 A Large-Scale Benchmark Dataset.csv"

def load_data(path=RAW_PATH, llms=LLMS):
    # the CSV is not utf-8, so read it as latin-1
    df = pd.read_csv(path, encoding="latin-1")
    # keep only rows written by the chosen LLMs
    df = df[df["Source"].isin(llms)]
    # rename columns to the names the whole team uses
    df = df.rename(columns={"Question": "input", "Answer": "output", "Source": "label"})
    # drop rows with a missing question or answer
    df = df.dropna(subset=["input", "output"])
    return df.reset_index(drop=True)