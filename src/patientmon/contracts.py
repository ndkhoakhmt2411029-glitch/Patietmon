"""Cấu trúc dữ liệu dùng chung giữa các khối. Các giai đoạn sau (tracking, pose...)
chỉ trao đổi với nhau qua những cấu trúc này."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Một hộp phát hiện. box = (x1, y1, x2, y2) tính bằng pixel."""

    box: tuple
    score: float
    cls: int


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    ok: bool
    fps: float = 0.0
    width: int = 0
    height: int = 0
    n_frames: int = 0

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class FrameRecord:
    """Một khung đã cắt từ video. rel_path tương đối so với thư mục frames."""

    video_id: str
    frame_idx: int
    time_sec: float
    rel_path: str


@dataclass(frozen=True)
class TrackedBox:
    """Một người đang được theo dõi ở một khung: ID ổn định giữa các khung."""

    track_id: int
    box: tuple
    score: float
    cls: int = 0


@dataclass
class SkeletonSequence:
    """Skeleton của mọi người được theo dõi trong một nguồn (video hoặc thư mục ảnh), dạng bảng dài:
    mỗi hàng là một (khung, người). frame_idx là chỉ số khung của NGUỒN; hai khung liên tiếp cách nhau `step`."""

    video_id: str
    fps: float  # FPS hiệu dụng sau khi lấy mẫu
    step: int
    width: int
    height: int
    frame_idx: np.ndarray  # (K,)
    track_id: np.ndarray  # (K,)
    keypoints: np.ndarray  # (K, 17, 3): x, y (pixel), độ tin cậy
    boxes: np.ndarray  # (K, 4)
    scores: np.ndarray  # (K,)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, video_id=self.video_id, fps=self.fps, step=self.step, width=self.width, height=self.height,
            frame_idx=self.frame_idx, track_id=self.track_id, keypoints=self.keypoints, boxes=self.boxes, scores=self.scores,
        )

    @classmethod
    def load(cls, path) -> "SkeletonSequence":
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["video_id"]), float(z["fps"]), int(z["step"]), int(z["width"]), int(z["height"]),
                       z["frame_idx"], z["track_id"], z["keypoints"], z["boxes"], z["scores"])
