"""TCN cơ sở (conv giãn nở theo thời gian) trên chuỗi khớp đã làm phẳng; dùng làm mốc so sánh."""
from __future__ import annotations

import torch.nn as nn

from .graph import NUM_JOINTS


class _Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # kernel lẻ để giữ nguyên độ dài
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(out_c), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Conv1d(out_c, out_c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(out_c))
        self.res = nn.Identity() if in_c == out_c else nn.Conv1d(in_c, out_c, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.res(x))


class TCN(nn.Module):
    def __init__(self, num_classes: int, in_features: int = NUM_JOINTS * 3, hidden: int = 128, layers: int = 4,
                 kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks, c_in = [], in_features
        for i in range(layers):
            blocks.append(_Block(c_in, hidden, 2 ** i, kernel, dropout))
            c_in = hidden
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):  # (N, T, V, C)
        n, t, v, c = x.shape
        return self.fc(self.blocks(x.reshape(n, t, v * c).transpose(1, 2)).mean(dim=-1))
