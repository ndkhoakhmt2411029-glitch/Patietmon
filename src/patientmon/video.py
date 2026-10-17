"""Đọc thông tin video và lấy mẫu khung hình."""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List

import cv2

from .contracts import VideoInfo

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def find_videos(root) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_text).strip("_")


def assign_ids(videos, root) -> dict:
    """ID ổn định, chỉ gồm ký tự ASCII (an toàn cho OpenCV trên Windows). Nhận cả file video lẫn thư mục ảnh.
    Báo lỗi nếu hai nguồn khác nhau trùng ID."""
    ids = {}
    for path in videos:
        rel = Path(path).relative_to(root)
        parts = (rel if Path(path).is_dir() else rel.with_suffix("")).parts or (Path(root).name,)
        ids[path] = _slug("__".join(parts))
    duplicated = [k for k, n in Counter(ids.values()).items() if n > 1]
    if duplicated:
        raise ValueError(f"Hai video khác nhau cho cùng video_id {duplicated}; hãy đổi tên file.")
    return ids


def probe(path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return VideoInfo(Path(path), ok=False)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return VideoInfo(Path(path), fps > 0 and n > 0 and w > 0 and h > 0, fps, w, h, n)
    finally:
        cap.release()


def probe_report(infos: List[VideoInfo]) -> str:
    good = [i for i in infos if i.ok]
    lines = [
        f"Số video: {len(infos)} (đọc được {len(good)}, lỗi {len(infos) - len(good)})",
        f"Tổng thời lượng: {sum(i.duration for i in good) / 3600:.2f} giờ",
        f"FPS gặp: {sorted({round(i.fps) for i in good})}",
        f"Độ phân giải: {dict(Counter(f'{i.width}x{i.height}' for i in good))}",
    ]
    lines += [f"  LỖI: {i.path}" for i in infos if not i.ok]
    return "\n".join(lines)


def write_probe_csv(infos: List[VideoInfo], root, csv_path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "ok", "fps", "width", "height", "n_frames", "duration_sec"])
        for i in infos:
            writer.writerow([Path(i.path).relative_to(root), i.ok, i.fps, i.width, i.height, i.n_frames, round(i.duration, 2)])
