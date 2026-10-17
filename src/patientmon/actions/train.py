"""Huấn luyện bộ phân loại hành động trên cửa sổ skeleton. Mọi mô hình (ST-GCN++, LSTM, GRU, TCN) dùng chung
một vòng huấn luyện, nên bảng so sánh chạy trên cùng dữ liệu và cùng cách chia."""
from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import Config
from .data import WINDOWS_FILE, load_windows
from .metrics import confusion_matrix, report
from .models import build_model
from .models.graph import LEFT_RIGHT, NUM_JOINTS

log = logging.getLogger(__name__)
_FLIP = list(range(NUM_JOINTS))
for _l, _r in LEFT_RIGHT:
    _FLIP[_l], _FLIP[_r] = _r, _l


def model_params(cfg: Config) -> dict:
    t = cfg.train_action
    return {"base_channels": t.base_channels, "layers": t.layers, "rnn_layers": t.rnn_layers, "hidden": t.hidden,
            "tcn_layers": t.tcn_layers, "dropout": t.dropout}


def augment(x):
    """Tăng cường dữ liệu nhẹ trên (N, T, V, 3) đã chuẩn hóa: xoay +-10 độ, co giãn +-10%, lật ngang (đổi khớp
    trái/phải), nhiễu nhỏ. Chỉ dùng khi huấn luyện."""
    import torch

    n, dev = x.size(0), x.device
    ang = (torch.rand(n, 1, 1, device=dev) * 2 - 1) * (math.pi / 18)
    scale = 1 + (torch.rand(n, 1, 1, device=dev) * 2 - 1) * 0.1
    flip = torch.rand(n, 1, 1, device=dev) < 0.5
    px, py = torch.where(flip, -x[..., 0], x[..., 0]), x[..., 1]
    cos, sin = torch.cos(ang), torch.sin(ang)
    xy = torch.stack([(cos * px - sin * py) * scale, (sin * px + cos * py) * scale], dim=-1)
    xy = xy + torch.randn_like(xy) * 0.01
    full = torch.cat([xy, x[..., 2:]], dim=-1)
    return torch.where(flip.unsqueeze(-1), full[:, :, _FLIP, :], full)


def predict_probs(model, X: np.ndarray, stream: str, device, num_classes: int, batch: int = 64) -> np.ndarray:
    import torch

    from .models.features import to_stream

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = to_stream(torch.from_numpy(X[i : i + batch]).to(device), stream)
            outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, num_classes), dtype=np.float32)


def _new_run_dir(base: Path) -> Path:
    path, n = base, 2
    while path.exists():
        path, n = base.with_name(f"{base.name}_{n}"), n + 1
    path.mkdir(parents=True)
    return path


def train_action(cfg: Config, windows_path=None, model_name: str = None) -> Path:
    """Huấn luyện và trả về đường dẫn checkpoint tốt nhất (theo macro-F1 trên tập val)."""
    from ..compat import import_torch

    torch = import_torch()
    import torch.nn as nn

    from .models.features import to_stream

    t = cfg.train_action
    name = model_name or t.model
    data = load_windows(windows_path or cfg.paths.resolve("windows") / WINDOWS_FILE)
    n_cls = len(data.classes)
    random.seed(t.seed), np.random.seed(t.seed), torch.manual_seed(t.seed)
    device = torch.device(t.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tr, va = data.split == "train", data.split == "val"
    if not tr.any() or not va.any():
        raise ValueError("Cần có cả tập train và val trong windows.npz")
    X_tr, y_tr = torch.from_numpy(data.X[tr]), torch.from_numpy(data.y[tr]).long()
    X_va, y_va = data.X[va], data.y[va]

    params = model_params(cfg)
    model = build_model(name, n_cls, params).to(device)
    weight = None
    if t.class_weights:
        counts = np.bincount(data.y[tr], minlength=n_cls).astype(float)
        w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_cls, 0.0)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t.epochs))

    run_dir = _new_run_dir(cfg.paths.resolve("runs") / "action" / (t.name or name))
    best_path, best, bad = run_dir / "best.pt", -1.0, 0
    log_file = (run_dir / "log.csv").open("w", encoding="utf-8", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_macro_f1", "val_accuracy", "lr"])
    (run_dir / "run_info.json").write_text(json.dumps(
        {"model": name, "stream": t.stream, "device": str(device), "torch": torch.__version__, "config": asdict(cfg)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        for epoch in range(1, t.epochs + 1):
            model.train()
            perm, total, count = torch.randperm(len(X_tr)), 0.0, 0
            for i in range(0, len(perm), t.batch):
                idx = perm[i : i + t.batch]
                if len(idx) < 2:  # BatchNorm cần ít nhất 2 mẫu
                    continue
                xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
                if t.augment:
                    xb = augment(xb)
                loss = loss_fn(model(to_stream(xb, t.stream)), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, count = total + loss.item() * len(idx), count + len(idx)
            sched.step()
            probs = predict_probs(model, X_va, t.stream, device, n_cls)
            rep = report(confusion_matrix(y_va, probs.argmax(1), n_cls))
            writer.writerow([epoch, round(total / max(count, 1), 5), round(rep["macro_f1"], 4), round(rep["accuracy"], 4),
                             opt.param_groups[0]["lr"]])
            log_file.flush()
            log.info("epoch %d: loss %.4f, val macro-F1 %.4f", epoch, total / max(count, 1), rep["macro_f1"])
            if rep["macro_f1"] > best:
                best, bad = rep["macro_f1"], 0
                torch.save({"model": name, "state_dict": model.state_dict(), "classes": data.classes, "params": params,
                            "stream": t.stream, "window_frames": int(data.X.shape[1]), "fps": data.fps,
                            "val_macro_f1": best}, best_path)
            else:
                bad += 1
                if bad >= t.patience:
                    log.info("Dừng sớm ở epoch %d (không cải thiện %d epoch)", epoch, t.patience)
                    break
    finally:
        log_file.close()
    return best_path
