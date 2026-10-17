"""Chạy chuỗi Giai đoạn 2-4 trên một nguồn: phát hiện người -> theo dõi -> pose -> skeleton theo từng người."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

from ..contracts import SkeletonSequence

log = logging.getLogger(__name__)


def extract_skeletons(
    source, detector, tracker_factory: Callable[[float], object], pose, target_fps: Optional[float] = None,
    det_batch: int = 8, person_class: int = 0,
) -> SkeletonSequence:
    """tracker_factory(fps) tạo bộ theo dõi mới cho mỗi nguồn (không giữ ID giữa các nguồn).
    Nếu target_fps nhỏ hơn FPS gốc, lấy mẫu xuống bằng bước nguyên `step` (FPS hiệu dụng = fps gốc / step)."""
    step = 1 if not target_fps else max(1, round(source.fps / target_fps))
    fps = source.fps / step
    tracker = tracker_factory(fps)
    rows: List[tuple] = []
    size = [0, 0]

    def process(batch):
        for frame, dets in zip(batch, detector.detect([f.image for f in batch])):
            size[1], size[0] = frame.image.shape[:2]
            tracks = tracker.update([d for d in dets if d.cls == person_class])
            if not tracks:
                continue
            for t, kp in zip(tracks, pose.estimate(frame.image, [t.box for t in tracks])):
                rows.append((frame.idx, t.track_id, kp, t.box, t.score))

    batch = []
    for frame in source.frames(step):
        batch.append(frame)
        if len(batch) == det_batch:
            process(batch)
            batch = []
    if batch:
        process(batch)
    log.info("%s: %d hàng (khung, người), %d track", source.id, len(rows), len({r[1] for r in rows}))
    return SkeletonSequence(
        source.id, fps, step, size[0], size[1],
        np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[2] for r in rows], dtype=np.float32).reshape(-1, 17, 3),
        np.array([r[3] for r in rows], dtype=np.float32).reshape(-1, 4), np.array([r[4] for r in rows], dtype=np.float32),
    )
