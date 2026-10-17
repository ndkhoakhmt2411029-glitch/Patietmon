"""ByteTrack tự cài đặt (Zhang và cộng sự, 2022), không phụ thuộc thư viện ngoài ngoài numpy/scipy.

Ý tưởng: ghép hộp điểm cao với các track trước, rồi dùng hộp điểm THẤP để cứu các track còn lại
(người bị che một phần thường cho hộp điểm thấp). Khác biệt nhỏ so với bản gốc: bước ghép dùng
scipy.optimize.linear_sum_assignment rồi loại cặp vượt ngưỡng, thay cho lapjv.
Giống bản gốc: track mới chỉ được xuất ra từ khung thứ hai (khi đã được xác nhận), trừ khung đầu tiên."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..contracts import Detection, TrackedBox
from .kalman import KalmanFilter

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


def _to_xyah(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], dtype=float)


class _Track:
    def __init__(self, det: Detection):
        self.box0, self.score, self.cls = det.box, det.score, det.cls
        self.mean = self.cov = None
        self.state, self.is_activated = NEW, False
        self.track_id, self.start_frame, self.end_frame = 0, 0, 0

    @property
    def box(self) -> np.ndarray:
        if self.mean is None:
            return np.asarray(self.box0, dtype=float)
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self, kf: KalmanFilter) -> None:
        mean = self.mean.copy()
        if self.state != TRACKED:
            mean[7] = 0.0  # track đang mất: không ngoại suy chiều cao
        self.mean, self.cov = kf.predict(mean, self.cov)

    def activate(self, kf: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.mean, self.cov = kf.initiate(_to_xyah(self.box0))
        self.track_id, self.state = track_id, TRACKED
        self.is_activated = frame_id == 1
        self.start_frame = self.end_frame = frame_id

    def update(self, kf: KalmanFilter, det: Detection, frame_id: int) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, _to_xyah(det.box))
        self.state, self.is_activated = TRACKED, True
        self.score, self.cls, self.end_frame = det.score, det.cls, frame_id


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _associate(tracks: List[_Track], dets: Sequence[Detection], thresh: float) -> Tuple[list, list, list]:
    """Trả (các cặp (chỉ số track, chỉ số det), track chưa ghép, det chưa ghép)."""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    cost = 1.0 - _iou_matrix(np.array([t.box for t in tracks]), np.array([d.box for d in dets]))
    rows, cols = linear_sum_assignment(np.where(cost > thresh, 1e5, cost))  # cặp quá xa không làm lệch lời giải
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    matched_t, matched_d = {r for r, _ in matches}, {c for _, c in matches}
    return (matches, [i for i in range(len(tracks)) if i not in matched_t],
            [j for j in range(len(dets)) if j not in matched_d])


class ByteTracker:
    def __init__(self, high_thresh: float = 0.25, low_thresh: float = 0.1, new_thresh: float = 0.25,
                 buffer: int = 30, match_thresh: float = 0.8, frame_rate: float = 30.0):
        self.high, self.low, self.new_thresh, self.match_thresh = high_thresh, low_thresh, new_thresh, match_thresh
        self.max_time_lost = max(1, int(frame_rate / 30.0 * buffer))
        self.kf = KalmanFilter()
        self.reset()

    def reset(self) -> None:
        self.frame_id, self.next_id = 0, 1
        self.tracked: List[_Track] = []
        self.lost: List[_Track] = []

    def update(self, detections: Sequence[Detection]) -> List[TrackedBox]:
        self.frame_id += 1
        high = [d for d in detections if d.score >= self.high]
        low = [d for d in detections if self.low < d.score < self.high]
        unconfirmed = [t for t in self.tracked if not t.is_activated]
        pool = [t for t in self.tracked if t.is_activated] + self.lost
        for t in pool:
            t.predict(self.kf)

        # 1) hộp điểm cao <-> track đã xác nhận và track đang mất
        matches, u_pool, u_high = _associate(pool, high, self.match_thresh)
        for ti, di in matches:
            pool[ti].update(self.kf, high[di], self.frame_id)

        # 2) hộp điểm thấp <-> track còn lại đang được theo dõi
        remain = [pool[i] for i in u_pool if pool[i].state == TRACKED]
        matches, u_remain, _ = _associate(remain, low, 0.5)
        for ti, di in matches:
            remain[ti].update(self.kf, low[di], self.frame_id)
        for i in u_remain:
            remain[i].state = LOST

        # 3) track chưa xác nhận <-> hộp điểm cao chưa dùng
        left = [high[i] for i in u_high]
        matches, u_unconf, u_left = _associate(unconfirmed, left, 0.7)
        for ti, di in matches:
            unconfirmed[ti].update(self.kf, left[di], self.frame_id)
        for i in u_unconf:
            unconfirmed[i].state = REMOVED

        # 4) khởi tạo track mới từ hộp điểm cao chưa ghép được
        for j in u_left:
            if left[j].score >= self.new_thresh:
                t = _Track(left[j])
                t.activate(self.kf, self.frame_id, self.next_id)
                self.next_id += 1
                self.tracked.append(t)

        # 5) cập nhật danh sách, loại track mất quá lâu
        everyone = {id(t): t for t in self.tracked + self.lost}
        for t in everyone.values():
            if t.state == LOST and self.frame_id - t.end_frame > self.max_time_lost:
                t.state = REMOVED
        self.tracked = [t for t in everyone.values() if t.state == TRACKED]
        self.lost = [t for t in everyone.values() if t.state == LOST]
        return [TrackedBox(t.track_id, tuple(float(v) for v in t.box), float(t.score), int(t.cls))
                for t in self.tracked if t.is_activated]
