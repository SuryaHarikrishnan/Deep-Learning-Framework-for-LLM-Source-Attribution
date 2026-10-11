
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=6):
        super().__init__()

        # Convert token IDs into dense word representations
        self.embedding = nn.Embedding(
            vocab_size, 128, padding_idx=0
        )

        # Detect patterns of 3, 4, and 5 consecutive tokens
        self.convs = nn.ModuleList([
            nn.Conv1d(128, 100, kernel_size=k)
            for k in [3, 4, 5]
        ])

        # Reduce overfitting during training
        self.dropout = nn.Dropout(0.5)

        # Predict one of the six LLM families
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        # x: (batch, 256)
        x = self.embedding(x)

        # Conv1d expects (batch, channels, sequence_length)
        x = x.transpose(1, 2)

        # Apply convolution, ReLU, and max pooling
        features = [
            F.adaptive_max_pool1d(
                F.relu(conv(x)), 1
            ).squeeze(2)
            for conv in self.convs
        ]

        # Combine features from all filter sizes
        x = torch.cat(features, dim=1)

        x = self.dropout(x)
        return self.fc(x)
