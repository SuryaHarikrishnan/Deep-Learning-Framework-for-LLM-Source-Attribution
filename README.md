## Repository Structure

```text
Deep-Learning-Framework-for-LLM-Source-Attribution/
├── README.md
├── requirements.txt
├── .gitignore                  # ignore data/ and checkpoints
├── report/
│   └── report.pdf              # names of leader + members
├── data/
│   ├── README.md               # source, citation, how to download
│   ├── raw/                    # TXD-22 CSV (not committed)
│   └── processed/              # splits, vocab (not committed)
├── src/
│   ├── data/
│   │   ├── load.py             # read CSV, clean, filter classes
│   │   ├── split.py            # stratified train/val/test + domain splits
│   │   └── dataset.py          # tokenizer, vocab, PyTorch Dataset
│   ├── models/
│   │   ├── cnn.py              # TextCNN
│   │   ├── rnn.py              # BiLSTM
│   │   └── baselines.py        # TF-IDF + logistic regression
│   ├── train.py                # shared training loop
│   ├── evaluate.py             # accuracy, macro F1, confusion matrix
│   └── utils.py                # seeds, logging, config
├── experiments/
│   ├── rq1_output_only.py
│   ├── rq2_input_vs_output.py  # input only, output only, both
│   ├── rq3_cross_domain.py     # extra credit
│   └── rq4_feature_analysis.py # extra credit: length, vocab, formatting
├── configs/
│   ├── cnn.yaml
│   └── rnn.yaml
├── notebooks/
│   └── eda.ipynb               # class balance, length stats
├── results/
│   ├── figures/
│   └── tables/
└── scripts/
    └── run_all.sh
```
```
