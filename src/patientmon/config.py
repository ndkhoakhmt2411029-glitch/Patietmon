"""Đọc cấu hình YAML thành dataclass. Khóa lạ bị từ chối để bắt lỗi gõ sai sớm."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class PathsCfg:
    data_root: str = "data"
    videos: str = "videos"
    frames: str = "frames"
    labels_auto: str = "labels_auto"
    labels_reviewed: str = "labels_reviewed"
    yolo_dataset: str = "yolo"
    skeletons: str = "skeletons"
    windows: str = "windows"
    runs: str = "runs"
    meta_file: Optional[str] = None

    def resolve(self, name: str) -> Path:
        return Path(self.data_root) / getattr(self, name)

    def meta_path(self) -> Optional[Path]:
        return Path(self.data_root) / self.meta_file if self.meta_file else None


@dataclass
class FramesCfg:
    every_sec: float = 1.0
    min_diff: float = 2.0
    max_per_video: Optional[int] = None
    jpeg_quality: int = 95


@dataclass
class PrelabelCfg:
    weights: str = "yolov8x.pt"
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: Optional[str] = None
    batch: int = 16
    confident_score: float = 0.6
    class_map: Dict[int, int] = field(default_factory=lambda: {0: 0})


@dataclass
class DatasetCfg:
    names: List[str] = field(default_factory=lambda: ["person"])
    ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    seed: int = 0


@dataclass
class TrainCfg:
    weights: str = "yolov8s.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 15
    seed: int = 0
    device: Optional[str] = None
    freeze: Optional[int] = None
    name: str = "person_det"


@dataclass
class EvalCfg:
    baseline_weights: str = "yolov8s.pt"
    baseline_class_map: Dict[int, int] = field(default_factory=lambda: {0: 0})
    conf: float = 0.01
    report_conf: float = 0.25
    iou_match: float = 0.5
    imgsz: int = 640
    batch: int = 16
    target_class: int = 0



@dataclass
class SourceCfg:
    image_kind: str = "sequence"  # sequence: ảnh chụp liên tiếp | stills: ảnh đơn lẻ
    image_fps: Optional[float] = None  # bắt buộc với chuỗi ảnh liên tiếp


@dataclass
class DetectorCfg:  # bộ phát hiện dùng lúc trích skeleton (Giai đoạn 2)
    weights: str = "yolov8m.pt"  # hoặc best.pt đã fine-tune (khi đó đặt class_map: null)
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: Optional[str] = None
    batch: int = 8
    class_map: Optional[Dict[int, int]] = field(default_factory=lambda: {0: 0})


@dataclass
class TrackingCfg:  # ByteTrack (Giai đoạn 3)
    high_thresh: float = 0.25
    low_thresh: float = 0.1
    new_thresh: float = 0.25
    buffer: int = 30  # số khung (ở 30 FPS) giữ track bị mất
    match_thresh: float = 0.8


@dataclass
class PoseCfg:  # mô hình pose chạy trên từng khung người (Giai đoạn 4)
    weights: str = "yolov8m-pose.pt"
    imgsz: int = 256
    margin: float = 0.1  # nới khung người mỗi phía theo tỉ lệ kích thước
    device: Optional[str] = None


@dataclass
class SkeletonCfg:
    target_fps: Optional[float] = None  # null = giữ FPS gốc; số = lấy mẫu xuống FPS này


@dataclass
class ActionCfg:  # dữ liệu cho Giai đoạn 5
    classes: List[str] = field(default_factory=lambda: ["stand", "sit", "lie", "walk", "fall"])
    label_rules: List[Dict[str, str]] = field(default_factory=list)  # [{pattern: regex, label: lớp}], luật đầu tiên khớp thắng
    group_pattern: Optional[str] = None  # regex có 1 nhóm bắt mã người; null = mỗi đoạn là một nhóm
    window_frames: int = 32
    stride_frames: int = 16
    min_track_frames: int = 32  # bỏ track ngắn hơn mức này
    max_missing: float = 0.5  # bỏ cửa sổ có quá tỉ lệ này khung bị mất
    max_tracks_per_clip: Optional[int] = 1  # giữ N track dài nhất mỗi đoạn (null = giữ hết)
    conf_threshold: float = 0.2  # keypoint dưới ngưỡng này coi là mất và được nội suy
    ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    seed: int = 0


@dataclass
class TrainActionCfg:
    model: str = "stgcnpp"  # stgcnpp | lstm | gru | tcn
    stream: str = "joint"  # joint | bone | motion
    epochs: int = 60
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 0
    device: Optional[str] = None
    augment: bool = True
    class_weights: bool = True
    base_channels: int = 64  # stgcnpp
    layers: int = 10  # stgcnpp
    rnn_layers: int = 2  # lstm, gru
    hidden: int = 128  # lstm, gru, tcn
    tcn_layers: int = 4
    dropout: float = 0.2
    name: Optional[str] = None  # tên thư mục kết quả (mặc định: tên mô hình)


_SECTIONS = {
    "paths": PathsCfg,
    "frames": FramesCfg,
    "prelabel": PrelabelCfg,
    "dataset": DatasetCfg,
    "train": TrainCfg,
    "evaluate": EvalCfg,
    "source": SourceCfg,
    "detector": DetectorCfg,
    "tracking": TrackingCfg,
    "pose": PoseCfg,
    "skeleton": SkeletonCfg,
    "action": ActionCfg,
    "train_action": TrainActionCfg,
}


def _build(cls, data):
    data = data or {}
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Khóa không hợp lệ trong mục '{cls.__name__}': {sorted(unknown)}")
    return cls(**data)


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    frames: FramesCfg = field(default_factory=FramesCfg)
    source: SourceCfg = field(default_factory=SourceCfg)
    prelabel: PrelabelCfg = field(default_factory=PrelabelCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    evaluate: EvalCfg = field(default_factory=EvalCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    pose: PoseCfg = field(default_factory=PoseCfg)
    skeleton: SkeletonCfg = field(default_factory=SkeletonCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    train_action: TrainActionCfg = field(default_factory=TrainActionCfg)

    @classmethod
    def from_yaml(cls, path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"Mục cấu hình không hợp lệ: {sorted(unknown)}")
        return cls(**{name: _build(_SECTIONS[name], value) for name, value in raw.items()})


DetectionConfig = Config  # tên cũ, giữ để tương thích
