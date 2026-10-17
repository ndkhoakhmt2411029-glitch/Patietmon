"""Đánh giá và so sánh các checkpoint trên cùng một tập: mức cửa sổ, mức đoạn, tham số, tốc độ."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import WindowData
from .metrics import clip_level, confusion_matrix, report
from .models import build_model
from .train import predict_probs


def evaluate_checkpoint(ckpt_path, data: WindowData, split: str, device=None) -> dict:
    from ..compat import import_torch

    torch = import_torch()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = ckpt["classes"]
    if classes != data.classes:
        raise ValueError(f"Lớp của checkpoint {classes} khác lớp của dữ liệu {data.classes}")
    model = build_model(ckpt["model"], len(classes), ckpt["params"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    mask = data.split == split
    if not mask.any():
        raise ValueError(f"Không có mẫu nào trong tập '{split}'")
    X, y, keys = data.X[mask], data.y[mask], data.clip_key[mask]
    predict_probs(model, X[:4], ckpt["stream"], device, len(classes))  # khởi động trước khi đo giờ
    t0 = time.perf_counter()
    probs = predict_probs(model, X, ckpt["stream"], device, len(classes))
    ms = 1000 * (time.perf_counter() - t0) / len(X)
    window = report(confusion_matrix(y, probs.argmax(1), len(classes)))
    clip_true, clip_pred, _ = clip_level(probs, y, keys)
    clip = report(confusion_matrix(clip_true, clip_pred, len(classes)))
    return {"name": Path(ckpt_path).parent.name, "model": ckpt["model"], "stream": ckpt["stream"],
            "params": sum(p.numel() for p in model.parameters()), "windows": window, "clips": clip, "n_clips": len(clip_true),
            "ms_per_window": ms, "confusion": confusion_matrix(y, probs.argmax(1), len(classes)), "classes": classes}


def format_table(results: List[dict]) -> str:
    classes = results[0]["classes"]
    head = ["Mô hình", "Luồng", "Tham số (M)", "macro-F1", "Acc", "Acc mức đoạn", "ms/cửa sổ"] + [f"Recall {c}" for c in classes]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in results:
        w = r["windows"]
        row = [r["name"], r["stream"], f"{r['params'] / 1e6:.2f}", f"{w['macro_f1']:.3f}", f"{w['accuracy']:.3f}",
               f"{r['clips']['accuracy']:.3f} ({r['n_clips']} đoạn)", f"{r['ms_per_window']:.1f}"]
        row += [f"{v:.3f}" if s > 0 else "-" for v, s in zip(w["recall"], w["support"])]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_confusion(result: dict) -> str:
    classes, cm = result["classes"], result["confusion"]
    lines = [f"Ma trận nhầm lẫn của {result['name']} (hàng = thật, cột = dự đoán)", "\t" + "\t".join(classes)]
    lines += [f"{c}\t" + "\t".join(str(v) for v in row) for c, row in zip(classes, cm)]
    return "\n".join(lines)
