# CNN Model — Implementation and Results

## 1. Model Overview

We implemented a Convolutional Neural Network (CNN) to identify which of six large language model families generated a given text response. The model was trained and evaluated using the TXD-22 dataset.

## 2. CNN Architecture

The CNN uses the following components:

- **Embedding layer:** Converts token IDs into 128-dimensional word embeddings.
- **Convolutional layers:** Three parallel 1D convolutional filters with kernel sizes of 3, 4, and 5, each producing 100 feature maps.
- **Activation:** ReLU.
- **Pooling:** Global max pooling extracts the strongest feature from each convolutional filter.
- **Dropout:** 0.5 to reduce overfitting.
- **Fully connected layer:** Maps the combined 300 features to six output classes.

The model takes sequences of 256 token IDs and produces six classification scores.

## 3. Training Configuration

| Parameter | Value |
|---|---|
| Dataset | TXD-22 |
| Training samples | 14,448 |
| Validation samples | 1,794 |
| Testing samples | 1,758 |
| Input | LLM-generated answers |
| Batch size | 64 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-entropy |
| Random seed | 42 |

The shared training pipeline selected the model checkpoint with the highest validation accuracy before evaluating it on the test set.

## 4. Experimental Results

| Metric | Result |
|---|---|
| Best validation accuracy | 82.61% |
| Best epoch | 7 |
| Test accuracy | 81.40% |
| Test macro F1 | 81.35% |

## 5. Comparison with Baseline

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 86.30% | 86.30% |
| CNN | 81.40% | 81.35% |

The CNN achieved strong classification performance but scored approximately 4.9 percentage points below the traditional machine learning baseline.

This demonstrates that both traditional machine learning and deep learning approaches can identify stylistic and linguistic patterns associated with different LLM families.

## 6. Conclusion

The CNN was successfully implemented, trained, and evaluated on six LLM families. The results provide a deep learning comparison against the TF-IDF baseline and can be compared with the RNN results once available.