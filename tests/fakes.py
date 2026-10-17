"""Dữ liệu giả để kiểm thử mà không cần YOLO."""
import cv2
import numpy as np

from patientmon.contracts import Detection


class FakeDetector:
    """Trả về danh sách hộp định sẵn cho từng ảnh, theo thứ tự được gọi."""

    def __init__(self, per_image):
        self.per_image = per_image
        self.calls = 0

    def detect(self, frames):
        out = []
        for _ in frames:
            out.append(list(self.per_image[self.calls % len(self.per_image)]))
            self.calls += 1
        return out


def person(x1, y1, x2, y2, score=0.9, cls=0):
    return Detection((x1, y1, x2, y2), score, cls)


def write_video(path, n_frames=30, fps=10, size=(96, 64), moving=True):
    """Video AVI/MJPG tổng hợp: một hình vuông trắng, di chuyển hoặc đứng yên."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        x = (i * 4) % (size[0] - 20) if moving else 10
        cv2.rectangle(frame, (x, 20), (x + 20, 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


class FakePose:
    """Pose giả: đặt 17 khớp theo mẫu cố định bên trong hộp người, độ tin cậy 0.9."""

    def __init__(self):
        self.template = np.random.RandomState(0).rand(17, 2)

    def estimate(self, image, boxes):
        out = []
        for x1, y1, x2, y2 in boxes:
            kp = np.zeros((17, 3), np.float32)
            kp[:, 0] = x1 + self.template[:, 0] * (x2 - x1)
            kp[:, 1] = y1 + self.template[:, 1] * (y2 - y1)
            kp[:, 2] = 0.9
            out.append(kp)
        return out


def write_image_folder(folder, n=5, size=(48, 32)):
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((size[1], size[0], 3), (i * 40) % 255, np.uint8)
        cv2.imwrite(str(folder / f"img_{i + 1}.jpg"), img)


def make_sequence(video_id, n=40, track_ids=(1,), step=1, fps=10.0, drop=(), seed=0):
    """SkeletonSequence tổng hợp: mỗi track một dáng người nhẹ nhàng dao động; `drop` = các vị trí khung bị mất."""
    from patientmon.contracts import SkeletonSequence

    rng = np.random.RandomState(seed)
    base = rng.rand(17, 2) * 100 + 200
    rows = []
    for tid in track_ids:
        for p in range(n):
            if p in drop:
                continue
            kp = np.concatenate([base + 3 * np.sin(p / 3.0) + tid * 5, np.full((17, 1), 0.9)], axis=1)
            rows.append((p * step, tid, kp.astype(np.float32)))
    frame_idx = np.array([r[0] for r in rows], dtype=np.int64)
    return SkeletonSequence(video_id, fps, step, 640, 480, frame_idx, np.array([r[1] for r in rows], dtype=np.int64),
                            np.stack([r[2] for r in rows]), np.zeros((len(rows), 4), np.float32), np.ones(len(rows), np.float32))
