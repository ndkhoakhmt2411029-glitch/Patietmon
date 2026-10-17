"""Ước lượng pose kiểu top-down: cắt từng khung người rồi chạy YOLOv8-pose trên phần cắt.
Đây là một cài đặt của PoseEstimator; có thể thay bằng mô hình pose khác mà không đổi phần còn lại."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import NUM_KEYPOINTS


def crop_box(box: Sequence[float], width: int, height: int, margin: float):
    """Nới hộp mỗi phía theo tỉ lệ kích thước rồi cắt vào trong ảnh. Trả (x0, y0, x1, y1) nguyên."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x0, y0 = int(max(0, np.floor(x1 - mx))), int(max(0, np.floor(y1 - my)))
    x3, y3 = int(min(width, np.ceil(x2 + mx))), int(min(height, np.ceil(y2 + my)))
    return x0, y0, x3, y3


class YoloPoseEstimator:
    def __init__(self, weights: str = "yolov8m-pose.pt", imgsz: int = 256, margin: float = 0.1,
                 device: Optional[str] = None):
        from ..compat import import_yolo  # nhập muộn để các phần không cần mô hình chạy được khi chưa cài ultralytics

        self.model = import_yolo()(weights)
        self.imgsz, self.margin, self.device = imgsz, margin, device

    def estimate(self, image: np.ndarray, boxes: Sequence[Sequence[float]]) -> List[np.ndarray]:
        h, w = image.shape[:2]
        out = [np.zeros((NUM_KEYPOINTS, 3), np.float32) for _ in boxes]
        crops, slots = [], []
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = crop_box(box, w, h, self.margin)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                crops.append((image[y0:y1, x0:x1], x0, y0))
                slots.append(i)
        if not crops:
            return out
        results = self.model.predict([c for c, _, _ in crops], imgsz=self.imgsz, device=self.device, verbose=False)
        for slot, (crop, x0, y0), r in zip(slots, crops, results):
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            # chọn người nằm gần tâm phần cắt nhất (phần cắt có thể chứa cả người bên cạnh)
            centers = r.boxes.xywh.cpu().numpy()[:, :2]
            target = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
            k = int(np.argmin(np.linalg.norm(centers - target, axis=1)))
            xy = r.keypoints.xy.cpu().numpy()[k]
            conf = r.keypoints.conf.cpu().numpy()[k] if r.keypoints.conf is not None else np.ones(len(xy))
            out[slot] = np.concatenate([xy + np.array([x0, y0]), conf[:, None]], axis=1).astype(np.float32)
        return out
