"""Dựng bộ dữ liệu YOLO từ các nhãn ĐÃ DUYỆT. Chia tập theo video (hoặc theo nhóm, ví dụ theo người)
để các khung gần giống nhau của cùng một video không rơi vào cả train lẫn test."""
from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .contracts import FrameRecord
from .labels import label_path

INDEX_FILE = "index.csv"
_INDEX_FIELDS = ["split", "video_id", "group", "image", "label"]


def load_meta(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """Đọc CSV siêu dữ liệu video (cột bắt buộc: video_id; tùy chọn: group, condition, ...)."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} phải có cột 'video_id'")
        return {row["video_id"]: row for row in reader}


def group_of(video_id: str, meta: Dict[str, Dict[str, str]]) -> str:
    return meta.get(video_id, {}).get("group") or video_id


def split_groups(sizes: Dict[str, int], ratios: Dict[str, float], seed: int = 0) -> Dict[str, str]:
    """Gán mỗi nhóm vào đúng một tập, bám tỉ lệ theo số khung; tập nào cũng có ít nhất một nhóm."""
    names = [s for s, r in ratios.items() if r > 0]
    if len(sizes) < len(names):
        raise ValueError(f"Cần ít nhất {len(names)} video/nhóm để chia {names}, hiện chỉ có {len(sizes)}.")
    total_ratio = sum(ratios[s] for s in names)
    total = sum(sizes.values())
    target = {s: ratios[s] / total_ratio * total for s in names}
    filled = {s: 0 for s in names}
    count = {s: 0 for s in names}
    groups = sorted(sizes)
    random.Random(seed).shuffle(groups)
    assignment = {}
    for i, g in enumerate(groups):
        empty = [s for s in names if count[s] == 0]
        candidates = empty if len(empty) >= len(groups) - i else names  # đảm bảo không tập nào trống
        s = max(candidates, key=lambda n: target[n] - filled[n])
        assignment[g] = s
        filled[s] += sizes[g]
        count[s] += 1
    return assignment


def build_yolo_dataset(
    records: List[FrameRecord], frames_dir, labels_dir, out_dir, names: List[str], ratios: Dict[str, float], seed: int = 0,
    meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> dict:
    meta = meta or {}
    frames_dir, labels_dir, out_dir = Path(frames_dir), Path(labels_dir), Path(out_dir)
    reviewed = [r for r in records if label_path(labels_dir, r).exists()]
    if not reviewed:
        raise ValueError(f"Chưa có khung nào có nhãn đã duyệt trong {labels_dir}")

    sizes = Counter(group_of(r.video_id, meta) for r in reviewed)
    assignment = split_groups(dict(sizes), ratios, seed)

    if out_dir.exists():
        if not (out_dir / "data.yaml").exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} đã có nội dung không phải bộ dữ liệu do công cụ này tạo; từ chối xoá.")
        shutil.rmtree(out_dir)

    summary: Dict[str, dict] = {}
    index_rows = []
    for r in reviewed:
        group = group_of(r.video_id, meta)
        split = assignment[group]
        image_rel, label_rel = f"images/{split}/{Path(r.rel_path).name}", f"labels/{split}/{Path(r.rel_path).stem}.txt"
        for rel in (image_rel, label_rel):
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frames_dir / r.rel_path, out_dir / image_rel)
        shutil.copy2(label_path(labels_dir, r), out_dir / label_rel)
        index_rows.append([split, r.video_id, group, image_rel, label_rel])
        s = summary.setdefault(split, {"videos": set(), "frames": 0, "boxes": 0})
        s["videos"].add(r.video_id)
        s["frames"] += 1
        s["boxes"] += sum(1 for line in (out_dir / label_rel).read_text(encoding="utf-8").splitlines() if line.strip())

    data = {"path": str(out_dir.resolve()), "names": dict(enumerate(names))}
    data.update({split: f"images/{split}" for split in summary})
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with (out_dir / INDEX_FILE).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_INDEX_FIELDS)
        writer.writerows(index_rows)
    return {split: {**v, "videos": len(v["videos"])} for split, v in summary.items()}


def load_index(out_dir, split: Optional[str] = None) -> List[dict]:
    with (Path(out_dir) / INDEX_FILE).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]
