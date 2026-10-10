# Deep Learning Framework for LLM Source Attribution

CMPSC 448 midterm project: identify which LLM wrote a response, using CNN and RNN (BiLSTM) classifiers.
Dataset: TXD-22, 6 LLM families (ChatGPT, Claude, Gemini, Llama, Qwen, DeepSeek), 3,000 responses each.

## Quick start
```bash
pip install -r requirements.txt
python -m src.data.split          # create train/val/test splits
python -m src.models.baselines    # run the TF-IDF baseline
```
See `data/README.md` for how to get the dataset.

## Interface contract
- Data: `tr, va, te, vocab, labels = get_loaders(mode)`, with mode in `"input"`, `"output"` or `"both"`
- Model: takes a `(batch, 256)` tensor of token ids, returns `(batch, 6)` class scores
- Training: `run(model, tr, va, te, labels)` returns accuracy and macro F1
- Seed 42 everywhere. Splits are by question, so no question appears in two splits.

## Baseline results (TF-IDF + logistic regression, test set)
| Input | Accuracy | Macro F1 |
|-------|----------|----------|
| input only | 0.164 | 0.133 |
| output only | 0.863 | 0.863 |
| input + output | 0.858 | 0.858 |

## Team workflow
- Branch from `main` as `feature/<name>` and open a pull request to merge
- Edit only your own files to avoid conflicts
- Write short, clear commit messages