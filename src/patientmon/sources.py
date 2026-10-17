"""Nguồn khung hình thống nhất: file video, thư mục ảnh liên tiếp (như UP-Fall) hoặc thư mục ảnh đơn lẻ.
Mọi bước phía sau (cắt khung, trích skeleton) chỉ làm việc với giao diện này."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .video import assign_ids, find_videos, probe

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Frame:
    idx: int  # chỉ số khung trong nguồn (0, 1, 2, ...)
    time_sec: float
    image: np.ndarray  # BGR


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


class VideoSource:
    kind = "video"

    def __init__(self, path, source_id: str):
        info = probe(path)
        if not info.ok:
            raise ValueError(f"Không đọc được video: {path}")
        self.path, self.id, self.fps, self.n_frames = Path(path), source_id, info.fps, info.n_frames

    def frames(self, step: int = 1) -> Iterator[Frame]:
        """Trả mỗi `step` khung một lần (khung 0, step, 2*step, ...)."""
        cap = cv2.VideoCapture(str(self.path))
        idx = 0
        try:
            while True:
                if idx % step:  # grab nhanh hơn read vì không giải mã ra ảnh
                    if not cap.grab():
                        break
                else:
                    ok, image = cap.read()
                    if not ok:
                        break
                    yield Frame(idx, idx / self.fps, image)
                idx += 1
        finally:
            cap.release()


class ImageFolderSource:
    """Thư mục ảnh chụp liên tiếp, sắp theo tên (tự nhiên); cần biết FPS."""

    kind = "sequence"

    def __init__(self, folder, source_id: str, fps: Optional[float]):
        self.files = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=_natural_key)
        if not self.files:
            raise ValueError(f"Không có ảnh trong {folder}")
        if not fps or fps <= 0:
            raise ValueError("Chuỗi ảnh liên tiếp cần biết FPS: hãy đặt source.image_fps trong file cấu hình.")
        self.id, self.fps, self.n_frames = source_id, float(fps), len(self.files)

    def frames(self, step: int = 1) -> Iterator[Frame]:
        for idx in range(0, len(self.files), step):
            image = cv2.imread(str(self.files[idx]))
            if image is None:
                log.warning("Không đọc được ảnh %s", self.files[idx])
                continue
            yield Frame(idx, idx / self.fps, image)


class StillsSource(ImageFolderSource):
    """Ảnh đơn lẻ không liên quan về thời gian: lấy hết, FPS không có nghĩa."""

    kind = "stills"

    def __init__(self, folder, source_id: str):
        super().__init__(folder, source_id, fps=1.0)


def _image_dirs(root: Path) -> List[Path]:
    dirs = {p.parent for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    return sorted(dirs)


def discover_sources(root, image_kind: str = "sequence", image_fps: Optional[float] = None) -> list:
    """Tìm mọi file video và mọi thư mục có chứa ảnh dưới `root`."""
    root = Path(root)
    if image_kind not in ("sequence", "stills"):
        raise ValueError("source.image_kind phải là 'sequence' hoặc 'stills'")
    videos, folders = find_videos(root), _image_dirs(root)
    ids = assign_ids(videos + folders, root)
    sources = []
    for path in videos:
        try:
            sources.append(VideoSource(path, ids[path]))
        except ValueError as e:
            log.warning("Bỏ qua %s: %s", path, e)
    for folder in folders:
        sources.append(StillsSource(folder, ids[folder]) if image_kind == "stills"
                       else ImageFolderSource(folder, ids[folder], image_fps))
    return sources


def sample(source, every_sec: float, min_diff: float = 0.0, max_frames: Optional[int] = None) -> Iterator[Frame]:
    """Lấy mẫu thưa để gán nhãn: mỗi `every_sec` giây một khung (ảnh đơn lẻ: lấy hết).
    Nếu min_diff > 0, bỏ khung gần giống khung đã giữ trước đó (độ lệch xám trung bình < min_diff)."""
    step = 1 if source.kind == "stills" else max(1, round(source.fps * every_sec))
    last, kept = None, 0
    for frame in source.frames(step):
        if min_diff > 0 and source.kind != "stills":
            thumb = cv2.resize(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            if last is not None and float(np.abs(thumb - last).mean()) < min_diff:
                continue
            last = thumb
        yield frame
        kept += 1
        if max_frames and kept >= max_frames:
            break
