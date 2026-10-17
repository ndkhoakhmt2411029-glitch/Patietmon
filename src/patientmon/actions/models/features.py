"""Biến đổi đầu vào thành các luồng đặc trưng: joint (tọa độ), bone (vector xương), motion (hiệu giữa hai khung)."""
from __future__ import annotations

import torch

from .graph import bone_parents

STREAMS = ("joint", "bone", "motion")
_PARENTS = bone_parents()


def to_stream(x: torch.Tensor, stream: str) -> torch.Tensor:
    """x: (N, T, V, 3) gồm x, y, độ tin cậy. Trả tensor cùng hình dạng."""
    if stream == "joint":
        return x
    if stream == "bone":
        parents = torch.as_tensor(_PARENTS, device=x.device)
        xy = x[..., :2] - x[..., parents, :2]
        conf = torch.minimum(x[..., 2:], x[..., parents, 2:])
        return torch.cat([xy, conf], dim=-1)
    if stream == "motion":
        xy = x[:, 1:, :, :2] - x[:, :-1, :, :2]
        xy = torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=1)
        return torch.cat([xy, x[..., 2:]], dim=-1)
    raise ValueError(f"stream phải là một trong {STREAMS}, nhận được '{stream}'")
