"""Nhập thư viện nặng đúng lúc cần, kèm thông báo lỗi dễ hiểu khi chưa cài."""
from __future__ import annotations


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Cần cài ultralytics: pip install ultralytics (nên chạy trên Colab/Kaggle có GPU).") from e
    return YOLO


def import_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("Cần cài PyTorch: pip install torch (Colab và Kaggle đã có sẵn).") from e
    return torch
