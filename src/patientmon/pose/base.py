"""Giao diện chung của bộ ước lượng tư thế (pose)."""
from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np

NUM_KEYPOINTS = 17  # bố cục COCO


class PoseEstimator(Protocol):
    def estimate(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[np.ndarray]:
        """Với mỗi hộp người (x1, y1, x2, y2, pixel của `image`) trả mảng (17, 3) gồm x, y (pixel của `image`)
        và độ tin cậy. Người không tìm thấy keypoint trả mảng toàn 0."""
        ...
