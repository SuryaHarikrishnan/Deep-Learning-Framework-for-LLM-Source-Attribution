import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

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
        # Count non-padding tokens per sequence; packed LSTMs require CPU lengths.
        lengths = x.ne(0).sum(dim=1).clamp(min=1).cpu()
        # Map token IDs from (batch, sequence) to (batch, sequence, embedding).
        embedded = self.embedding(x)
        # Exclude trailing padding from LSTM processing, including for uneven lengths.
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        # Get the final hidden states from both directions of the bidirectional LSTM.
        _, (hidden, _) = self.lstm(packed)

        # Join the forward and backward features into one sequence representation.
        representation = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        # Return one unnormalized score per class for each input sequence.
        return self.classifier(representation)