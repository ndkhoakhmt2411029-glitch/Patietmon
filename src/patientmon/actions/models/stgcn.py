"""ST-GCN++ viết bằng PyTorch thuần, theo mô tả trong PYSKL (Duan và cộng sự, 2022):
đồ thị học được (khởi tạo từ đồ thị xương), khối GCN có nối tắt, và TCN nhiều nhánh (nhánh giãn nở, max-pool, 1x1).
LƯU Ý: đây là cài đặt lại theo mô tả, có thể không giống hệt bản gốc về chi tiết; hãy dùng như một mô hình
cùng họ khi so sánh. Mỗi mẫu là MỘT người: đầu vào (N, T, V, C)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .graph import NUM_JOINTS, spatial_adjacency


class GraphConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, adjacency: torch.Tensor):
        super().__init__()
        self.k = adjacency.size(0)
        self.A = nn.Parameter(adjacency.clone())  # học được, khởi tạo từ đồ thị xương
        self.conv = nn.Conv2d(in_c, out_c * self.k, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.down = nn.Identity() if in_c == out_c else nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # (N, C, T, V)
        n, _, t, v = x.shape
        y = self.conv(x).view(n, self.k, -1, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A).contiguous()
        return self.act(self.bn(y) + self.down(x))


class MSTCN(nn.Module):
    """TCN nhiều nhánh: hai nhánh conv giãn nở (1, 2), một nhánh max-pool, một nhánh 1x1."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, dilations=(1, 2), kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        branches = len(dilations) + 2
        mid = out_c // branches
        s = (stride, 1)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, (kernel, 1), stride=s, padding=(d * (kernel - 1) // 2, 0), dilation=(d, 1)),
                nn.BatchNorm2d(mid)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=s, padding=(1, 0)), nn.BatchNorm2d(mid)))
        rest = out_c - mid * (branches - 1)
        self.branches.append(nn.Sequential(nn.Conv2d(in_c, rest, 1, stride=s), nn.BatchNorm2d(rest)))
        self.transform = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(out_c, out_c, 1), nn.BatchNorm2d(out_c), nn.Dropout(dropout))

    def forward(self, x):
        return self.transform(torch.cat([b(x) for b in self.branches], dim=1))


class Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, adjacency: torch.Tensor, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, adjacency)
        self.tcn = MSTCN(out_c, out_c, stride, dropout=dropout)
        if in_c == out_c and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=(stride, 1)), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.tcn(self.gcn(x)) + self.res(x))


class STGCNpp(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, num_joints: int = NUM_JOINTS,
                 base_channels: int = 64, num_layers: int = 10, dropout: float = 0.2):
        super().__init__()
        adjacency = torch.tensor(spatial_adjacency(num_joints), dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        inflate = {round(num_layers * 0.4), round(num_layers * 0.7)}  # 10 lớp: tăng kênh và giảm nửa thời gian ở lớp 5 và 8
        blocks, c_in, c = [], in_channels, base_channels
        for i in range(num_layers):
            stride = 1
            if i in inflate and i > 0:
                c, stride = c * 2, 2
            blocks.append(Block(c_in, c, adjacency, stride, dropout=0.0))
            c_in = c
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):  # (N, T, V, C)
        n, t, v, c = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(n, v * c, t))
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.drop(x.mean(dim=(2, 3))))
