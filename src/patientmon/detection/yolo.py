"""Bộ phát hiện dựa trên Ultralytics YOLOv8."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..contracts import Detection


class YoloDetector:
    """class_map: {id lớp của mô hình -> id lớp mong muốn}. Lớp không có trong bảng bị bỏ.
    Mô hình COCO có sẵn dùng {0: 0} (chỉ giữ 'person'); mô hình đã fine-tune dùng None (giữ nguyên)."""

    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
                 device: Optional[str] = None, class_map: Optional[Dict[int, int]] = None):
        from ..compat import import_yolo  # nhập muộn: các bước không cần mô hình chạy được khi chưa cài ultralytics

        self.model = import_yolo()(weights)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.class_map = class_map

    def detect(self, frames: Sequence[np.ndarray]) -> List[List[Detection]]:
        if len(frames) == 0:
            return []
        results = self.model.predict(
            list(frames), conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
            classes=list(self.class_map) if self.class_map else None, verbose=False,
        )
        out = []
        for r in results:
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            dets = []
            for box, score, cls in zip(xyxy, scores, classes):
                cls = int(cls)
                if self.class_map is not None:
                    if cls not in self.class_map:
                        continue
                    cls = self.class_map[cls]
                dets.append(Detection(tuple(float(v) for v in box), float(score), cls))
            out.append(dets)
        return out
