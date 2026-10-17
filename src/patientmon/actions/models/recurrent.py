"""LSTM/GRU cơ sở trên chuỗi khớp đã làm phẳng; dùng làm mốc so sánh với ST-GCN++."""
from __future__ import annotations

import torch.nn as nn

from .graph import NUM_JOINTS


class RecurrentClassifier(nn.Module):
    def __init__(self, cell: str, num_classes: int, in_features: int = NUM_JOINTS * 3, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.norm = nn.LayerNorm(in_features)
        self.rnn = rnn(in_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):  # (N, T, V, C)
        n, t, v, c = x.shape
        out, _ = self.rnn(self.norm(x.reshape(n, t, v * c)))
        return self.fc(self.drop(out.mean(dim=1)))
