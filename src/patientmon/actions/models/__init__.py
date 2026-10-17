"""Các mô hình phân loại hành động. Nhập torch muộn để phần không cần torch (đồ thị, nhãn, chỉ số) vẫn dùng được."""
from __future__ import annotations

MODEL_NAMES = ("stgcnpp", "lstm", "gru", "tcn")


def build_model(name: str, num_classes: int, p: dict):
    """p: base_channels, layers, rnn_layers, hidden, tcn_layers, dropout (xem TrainActionCfg)."""
    if name == "stgcnpp":
        from .stgcn import STGCNpp
        return STGCNpp(num_classes, base_channels=p["base_channels"], num_layers=p["layers"], dropout=p["dropout"])
    if name in ("lstm", "gru"):
        from .recurrent import RecurrentClassifier
        return RecurrentClassifier(name, num_classes, hidden=p["hidden"], layers=p["rnn_layers"], dropout=p["dropout"])
    if name == "tcn":
        from .tcn import TCN
        return TCN(num_classes, hidden=p["hidden"], layers=p["tcn_layers"], dropout=p["dropout"])
    raise ValueError(f"Mô hình '{name}' không có; chọn một trong {MODEL_NAMES}")
