"""Dựng và đọc bộ dữ liệu cửa sổ skeleton cho Giai đoạn 5."""
from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from ..config import ActionCfg
from ..contracts import SkeletonSequence
from ..dataset import split_groups
from ..skeleton import normalize_window, select_tracks, track_windows
from .labeling import Labeler, make_group_fn

log = logging.getLogger(__name__)
WINDOWS_FILE, INDEX_FILE = "windows.npz", "index.csv"


@dataclass
class WindowData:
    X: np.ndarray  # (N, T, 17, 3) đã chuẩn hóa
    y: np.ndarray  # (N,)
    split: np.ndarray  # (N,) 'train' | 'val' | 'test'
    clip_key: np.ndarray  # (N,) "đoạn:track", dùng để gộp mức đoạn
    classes: List[str]
    fps: float


def build_window_dataset(skeleton_dir, out_dir, cfg: ActionCfg) -> dict:
    labeler, group_of = Labeler(cfg.classes, cfg.label_rules), make_group_fn(cfg.group_pattern)
    files = sorted(Path(skeleton_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"Không có file skeleton trong {skeleton_dir}")
    X, y, groups, keys, rows, fps_seen, skipped = [], [], [], [], [], set(), []
    for path in files:
        seq = SkeletonSequence.load(path)
        label = labeler.label_for(seq.video_id)
        if label is None:
            skipped.append(seq.video_id)
            continue
        fps_seen.add(round(seq.fps, 3))
        for tid in select_tracks(seq, cfg.min_track_frames, cfg.max_tracks_per_clip):
            starts, wins = track_windows(seq, tid, cfg.window_frames, cfg.stride_frames, cfg.max_missing)
            for start, win in zip(starts, wins):
                X.append(normalize_window(win, cfg.conf_threshold))
                y.append(label)
                groups.append(group_of(seq.video_id))
                keys.append(f"{seq.video_id}:{tid}")
                rows.append((seq.video_id, tid, int(start), cfg.classes[label]))
    if skipped:
        log.warning("%d nguồn không khớp luật nhãn nào nên bị bỏ qua, ví dụ: %s", len(skipped), skipped[:3])
    if not X:
        raise ValueError("Không tạo được cửa sổ nào: kiểm tra label_rules, min_track_frames và window_frames.")
    if len(fps_seen) > 1:
        log.warning("Các nguồn có FPS hiệu dụng khác nhau %s; nên đặt skeleton.target_fps để cùng nhịp.", sorted(fps_seen))

    assignment = split_groups(dict(Counter(groups)), cfg.ratios, cfg.seed)
    split = np.array([assignment[g] for g in groups])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / WINDOWS_FILE, X=np.stack(X).astype(np.float32), y=np.array(y, dtype=np.int64), split=split,
                        clip_key=np.array(keys), classes=np.array(cfg.classes), fps=np.float32(sorted(fps_seen)[0]))
    with (out_dir / INDEX_FILE).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "track_id", "start_pos", "label", "group", "split"])
        writer.writerows([*r, g, s] for r, g, s in zip(rows, groups, split))
    summary = {}
    for s in sorted(set(split)):
        in_split = split == s
        summary[s] = {"windows": int(in_split.sum()), "groups": len({g for g, m in zip(groups, in_split) if m}),
                      "per_class": {c: int(((np.array(y) == i) & in_split).sum()) for i, c in enumerate(cfg.classes)}}
    return summary


def load_windows(path) -> WindowData:
    path = Path(path)
    path = path / WINDOWS_FILE if path.is_dir() else path
    with np.load(path, allow_pickle=False) as z:
        return WindowData(z["X"], z["y"], z["split"], z["clip_key"], [str(c) for c in z["classes"]], float(z["fps"]))
