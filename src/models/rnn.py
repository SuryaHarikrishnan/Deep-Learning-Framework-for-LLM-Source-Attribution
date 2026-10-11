import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    # Classifies a sequence of token IDs into one of the source classes.
    def __init__(self, vocab_size, num_classes=6):
        super().__init__()
        embedding_dim = 128  # Size of each learned token representation.
        hidden_size = 128  # Number of hidden features in each LSTM direction.

        # Convert each token ID to a dense vector; keep padding (token ID 0)
        # represented by a fixed zero vector.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Read the sequence in both directions to capture left and right context.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # Regularize the combined sequence representation before classification.
        self.dropout = nn.Dropout(0.3)
        # Each direction contributes hidden_size features.
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        raise NotImplementedError