"""Đánh giá bộ phát hiện độc lập với thư viện huấn luyện: dùng chung giao diện Detector,
nên so sánh được mô hình có sẵn, mô hình fine-tune và các bộ phát hiện khác sau này."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .contracts import Detection
from .labels import read_labels


@dataclass
class Metrics:
    n_images: int
    n_gt: int
    n_pred: int          # số dự đoán có điểm >= report_conf
    tp: int
    precision: float
    recall: float
    f1: float
    ap50: float          # AP tại ngưỡng IoU dùng để ghép (mặc định 0.5)
    ms_per_image: float  # gần đúng, gồm cả lần gọi đầu tiên


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_image(preds: List[Detection], gts: List[Detection], iou_thr: float) -> List[Tuple[float, bool]]:
    """Ghép tham lam theo điểm giảm dần. Trả (điểm, là-TP) cho từng dự đoán."""
    used, out = set(), []
    for p in sorted(preds, key=lambda d: -d.score):
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if j not in used and iou(p.box, g.box) >= best_iou:
                best, best_iou = j, iou(p.box, g.box)
        if best >= 0:
            used.add(best)
        out.append((p.score, best >= 0))
    return out


def average_precision(scores, tps, n_gt: int) -> float:
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    tp = np.asarray(tps, dtype=float)[order]
    tpc, fpc = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(
    detector, samples: List[Tuple[Path, Path]], target_class: int = 0, iou_thr: float = 0.5,
    report_conf: float = 0.25, batch_size: int = 16,
) -> Metrics:
    """samples = [(ảnh, file nhãn YOLO)]. Chỉ xét lớp target_class.
    Detector nên chạy với conf thấp (<= report_conf) để AP đủ đường cong."""
    scored: List[Tuple[float, bool]] = []
    n_gt = n_img = 0
    elapsed = 0.0
    for i in range(0, len(samples), batch_size):
        batch = [(cv2.imread(str(img)), lab) for img, lab in samples[i : i + batch_size]]
        batch = [(im, lab) for im, lab in batch if im is not None]
        t0 = time.perf_counter()
        preds = detector.detect([im for im, _ in batch])
        elapsed += time.perf_counter() - t0
        for (im, lab), dets in zip(batch, preds):
            h, w = im.shape[:2]
            gts = [g for g in read_labels(lab, w, h) if g.cls == target_class]
            n_gt += len(gts)
            scored += match_image([d for d in dets if d.cls == target_class], gts, iou_thr)
            n_img += 1
    scores = np.array([s for s, _ in scored], dtype=float)
    flags = np.array([f for _, f in scored], dtype=bool)
    selected = scores >= report_conf
    tp, n_pred = int(flags[selected].sum()), int(selected.sum())
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(n_img, n_gt, n_pred, tp, precision, recall, f1, average_precision(scores, flags, n_gt),
                   1000 * elapsed / max(n_img, 1))


def evaluate_grouped(detector, rows: List[dict], out_dir, key: Callable[[dict], str], **kwargs) -> Dict[str, Metrics]:
    """Đánh giá riêng theo nhóm (ví dụ theo điều kiện ánh sáng). rows lấy từ dataset.load_index."""
    groups: Dict[str, List[Tuple[Path, Path]]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append((Path(out_dir) / r["image"], Path(out_dir) / r["label"]))
    return {name: evaluate(detector, samples, **kwargs) for name, samples in sorted(groups.items())}


def format_table(results: Dict[str, Metrics]) -> str:
    lines = ["| Nhóm | Ảnh | GT | Precision | Recall | F1 | AP50 | ms/ảnh |", "|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m.n_images} | {m.n_gt} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.ap50:.3f} | {m.ms_per_image:.1f} |")
    return "\n".join(lines)
