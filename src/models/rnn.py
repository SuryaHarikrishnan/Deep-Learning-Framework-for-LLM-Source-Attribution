import torch.nn as nn

class BiLSTM(nn.Module):
    # Person C implements this
    # input: LongTensor (batch, 256) of token ids
    # output: (batch, num_classes) class scores
    def __init__(self, vocab_size, num_classes=6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError