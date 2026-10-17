"""Fine-tune YOLOv8 bằng Ultralytics và lưu kèm thông tin tái lập (phiên bản thư viện, cấu hình)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import DetectionConfig


def train_detector(cfg: DetectionConfig, data_yaml: Path) -> Path:
    """Huấn luyện và trả về đường dẫn trọng số tốt nhất (best.pt)."""
    from ..compat import import_yolo

    YOLO = import_yolo()
    import ultralytics

    t = cfg.train
    args = dict(
        data=str(Path(data_yaml).resolve()), epochs=t.epochs, imgsz=t.imgsz, batch=t.batch, patience=t.patience,
        seed=t.seed, device=t.device, project=str(cfg.paths.resolve("runs").resolve()), name=t.name,
    )
    if t.freeze:
        args["freeze"] = t.freeze
    model = YOLO(t.weights)
    model.train(**args)
    run_dir = Path(model.trainer.save_dir)
    info = {"ultralytics": ultralytics.__version__, "config": asdict(cfg)}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir / "weights" / "best.pt"
