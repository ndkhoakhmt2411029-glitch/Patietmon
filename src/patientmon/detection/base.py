"""Giao diện chung của bộ phát hiện. Thay YOLOv8 bằng mô hình khác chỉ cần cài đặt lại lớp này."""
from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np

from ..contracts import Detection


class Detector(Protocol):
    def detect(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Nhận danh sách ảnh BGR, trả danh sách hộp phát hiện cho từng ảnh (cùng thứ tự)."""
        ...
