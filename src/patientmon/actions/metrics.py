"""Chỉ số phân loại (chỉ dùng numpy)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Hàng = nhãn thật, cột = dự đoán."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def report(cm: np.ndarray) -> Dict[str, object]:
    """Precision, recall, F1 từng lớp; macro-F1 (chỉ tính các lớp có mẫu thật); accuracy."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {
        "precision": precision, "recall": recall, "f1": f1, "support": support.astype(int),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
    }


def clip_level(probs: np.ndarray, y: np.ndarray, keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Gộp các cửa sổ của cùng một (đoạn, người) bằng cách lấy trung bình xác suất.
    Trả (nhãn thật, nhãn dự đoán, khóa) cho từng đoạn."""
    order: Dict[str, List[int]] = {}
    for i, k in enumerate(keys):
        order.setdefault(k, []).append(i)
    names = sorted(order)
    truth = np.array([y[order[k][0]] for k in names])
    pred = np.array([probs[order[k]].mean(axis=0).argmax() for k in names])
    return truth, pred, names
