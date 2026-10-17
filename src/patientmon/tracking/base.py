"""Giao diện chung của bộ theo dõi."""
from __future__ import annotations

from typing import List, Protocol, Sequence

from ..contracts import Detection, TrackedBox


class Tracker(Protocol):
    def update(self, detections: Sequence[Detection]) -> List[TrackedBox]:
        """Nhận các hộp phát hiện của MỘT khung (theo thứ tự thời gian), trả các người đang được theo dõi."""
        ...
