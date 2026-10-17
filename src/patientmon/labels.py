"""Nhãn định dạng YOLO (một file .txt mỗi ảnh: `lớp cx cy w h`, toạ độ chuẩn hoá 0-1) và
bước tạo nhãn nháp bằng mô hình có sẵn."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

import cv2

from .contracts import Detection, FrameRecord

log = logging.getLogger(__name__)
QUEUE_FILE = "review_queue.csv"
QUEUE_FIELDS = ["rel_path", "n_boxes", "min_score", "status"]
_PRIORITY = {"empty": 0, "uncertain": 1, "confident": 2}


def label_path(labels_dir: Path, rec: FrameRecord) -> Path:
    return Path(labels_dir) / Path(rec.rel_path).with_suffix(".txt")


def write_labels(path: Path, dets: List[Detection], width: int, height: int) -> None:
    lines = []
    for d in dets:
        x1, y1, x2, y2 = d.box
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"{d.cls} {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def read_labels(path: Path, width: int, height: int) -> List[Detection]:
    dets = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:  # bỏ dòng trống hoặc định dạng khác (ví dụ đa giác)
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        dets.append(Detection(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height), 1.0, cls))
    return dets


def _status(dets: List[Detection], confident_score: float) -> str:
    if not dets:
        return "empty"
    return "uncertain" if min(d.score for d in dets) < confident_score else "confident"


def _read_queue(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sort_queue(path: Path) -> None:
    rows = _read_queue(path)
    rows.sort(key=lambda r: (_PRIORITY[r["status"]], float(r["min_score"] or 0)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prelabel(detector, records: List[FrameRecord], frames_dir, labels_dir, batch_size: int = 16, confident_score: float = 0.6) -> Path:
    """Tạo nhãn nháp cho từng khung và hàng đợi duyệt (khung trống/không chắc xếp trước).
    Chạy lại được: khung đã có trong hàng đợi sẽ được bỏ qua."""
    frames_dir, labels_dir = Path(frames_dir), Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    queue = labels_dir / QUEUE_FILE
    done = {r["rel_path"] for r in _read_queue(queue)}
    todo = [r for r in records if r.rel_path not in done]
    is_new = not queue.exists()
    with queue.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
        if is_new:
            writer.writeheader()
        for i in range(0, len(todo), batch_size):
            recs, images = [], []
            for r in todo[i : i + batch_size]:
                img = cv2.imread(str(frames_dir / r.rel_path))
                if img is None:
                    log.warning("Không đọc được ảnh %s", r.rel_path)
                    continue
                recs.append(r)
                images.append(img)
            for r, img, dets in zip(recs, images, detector.detect(images)):
                h, w = img.shape[:2]
                write_labels(label_path(labels_dir, r), dets, w, h)
                writer.writerow(
                    {
                        "rel_path": r.rel_path,
                        "n_boxes": len(dets),
                        "min_score": f"{min(d.score for d in dets):.3f}" if dets else "",
                        "status": _status(dets, confident_score),
                    }
                )
            f.flush()
            log.info("Nhãn nháp: %d/%d khung", min(i + batch_size, len(todo)), len(todo))
    _sort_queue(queue)
    return queue
