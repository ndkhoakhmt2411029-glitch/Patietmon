"""Cắt khung hình để gán nhãn, từ video hoặc thư mục ảnh. Mỗi nguồn có file frames.csv riêng; file này vừa
là chỉ mục vừa là dấu "đã xong", nên chạy lại sau khi Colab ngắt sẽ bỏ qua các nguồn đã cắt."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

import cv2

from .config import FramesCfg, SourceCfg
from .contracts import FrameRecord
from .sources import discover_sources, sample

log = logging.getLogger(__name__)
INDEX_NAME = "frames.csv"
_FIELDS = ["frame_idx", "time_sec", "rel_path"]


def _read_index(frames_dir: Path, video_id: str) -> List[FrameRecord]:
    with (frames_dir / video_id / INDEX_NAME).open(encoding="utf-8", newline="") as f:
        return [FrameRecord(video_id, int(r["frame_idx"]), float(r["time_sec"]), r["rel_path"]) for r in csv.DictReader(f)]


def _write_index(frames_dir: Path, video_id: str, records: List[FrameRecord]) -> None:
    with (frames_dir / video_id / INDEX_NAME).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_FIELDS)
        writer.writerows([r.frame_idx, r.time_sec, r.rel_path] for r in records)


def extract_source(source, frames_dir: Path, cfg: FramesCfg) -> List[FrameRecord]:
    vid = source.id
    if (frames_dir / vid / INDEX_NAME).exists():
        return _read_index(frames_dir, vid)
    (frames_dir / vid).mkdir(parents=True, exist_ok=True)
    records = []
    for frame in sample(source, cfg.every_sec, cfg.min_diff, cfg.max_per_video):
        rel = f"{vid}/{vid}_f{frame.idx:07d}.jpg"
        if not cv2.imwrite(str(frames_dir / rel), frame.image, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]):
            raise OSError(f"Không ghi được ảnh {frames_dir / rel}")
        records.append(FrameRecord(vid, frame.idx, round(frame.time_sec, 3), rel))
    _write_index(frames_dir, vid, records)
    return records


def extract_all(videos_dir, frames_dir, cfg: FramesCfg, source_cfg: SourceCfg = None) -> List[FrameRecord]:
    source_cfg = source_cfg or SourceCfg()
    frames_dir = Path(frames_dir)
    sources = discover_sources(videos_dir, source_cfg.image_kind, source_cfg.image_fps)
    if not sources:
        raise FileNotFoundError(f"Không tìm thấy video hoặc thư mục ảnh nào trong {videos_dir}")
    records: List[FrameRecord] = []
    for n, source in enumerate(sources, 1):
        recs = extract_source(source, frames_dir, cfg)
        log.info("[%d/%d] %s (%s): %d khung", n, len(sources), source.id, source.kind, len(recs))
        records += recs
    return records


def load_manifest(frames_dir) -> List[FrameRecord]:
    frames_dir = Path(frames_dir)
    return [r for idx in sorted(frames_dir.glob(f"*/{INDEX_NAME}")) for r in _read_index(frames_dir, idx.parent.name)]
