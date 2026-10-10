# Data

## Dataset
TXD-22: A Large-Scale Benchmark Dataset (66,000 rows with columns Question, Answer, Source).

We use 6 LLM families, with 3,000 examples each (18,000 total):
ChatGPT, Claude, Gemini, Llama, Qwen, DeepSeek.

Column mapping used in code:
- Question -> `input`
- Answer -> `output`
- Source -> `label`

## Setup
1. Download the dataset and place the CSV in `data/raw/`
   (file name: `TXD-22 A Large-Scale Benchmark Dataset.csv`, latin-1 encoded).
2. Run `python -m src.data.split` from the repo root.
   This creates `train.csv`, `val.csv` and `test.csv` in `data/processed/`.

## Splits
Roughly 80/10/10 (about 14,400 train, 1,800 val, 1,800 test), seed 42.
Split by question: the same question is asked to most LLMs, so all rows
for a question stay in one split to prevent leakage.
Data files are not committed to git.
