"""Chọn track chính và cắt chuỗi skeleton thành các cửa sổ trượt có độ dài cố định."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..contracts import SkeletonSequence


def select_tracks(seq: SkeletonSequence, min_frames: int, max_tracks: Optional[int]) -> List[int]:
    """Giữ các track đủ dài (tính theo khoảng thời gian từ khung đầu tới khung cuối), ưu tiên track nhiều khung nhất."""
    found = []
    for tid in np.unique(seq.track_id):
        idx = seq.frame_idx[seq.track_id == tid] // seq.step
        if idx.max() - idx.min() + 1 >= min_frames:
            found.append((len(idx), int(tid)))
    found.sort(reverse=True)
    return [tid for _, tid in found[:max_tracks]]


def track_windows(seq: SkeletonSequence, track_id: int, window: int, stride: int, max_missing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Trả (vị trí bắt đầu của từng cửa sổ, mảng (N, window, 17, 3) chưa chuẩn hóa).
    Khung mà track bị mất được điền 0 (độ tin cậy 0, sẽ được nội suy khi chuẩn hóa);
    cửa sổ có tỉ lệ khung mất vượt max_missing bị bỏ."""
    mask = seq.track_id == track_id
    pos = seq.frame_idx[mask] // seq.step
    order = np.argsort(pos)
    pos, kp = pos[order], seq.keypoints[mask][order]
    length = int(pos[-1] - pos[0] + 1)
    dense = np.zeros((length, 17, 3), dtype=np.float32)
    present = np.zeros(length, dtype=bool)
    dense[pos - pos[0]], present[pos - pos[0]] = kp, True
    starts = [s for s in range(0, length - window + 1, stride) if 1.0 - present[s : s + window].mean() <= max_missing]
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty((0, window, 17, 3), dtype=np.float32)
    return np.array(starts) + int(pos[0]), np.stack([dense[s : s + window] for s in starts])
