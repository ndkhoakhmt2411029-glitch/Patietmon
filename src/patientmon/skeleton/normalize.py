"""Chuẩn hóa skeleton. MỘT hàm duy nhất, dùng chung cho huấn luyện và chạy thật để hai bên không lệch nhau."""
from __future__ import annotations

import warnings

import numpy as np

HIPS, SHOULDERS = (11, 12), (5, 6)  # chỉ số COCO


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanmedian(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a)


def normalize_window(keypoints: np.ndarray, conf_threshold: float = 0.2) -> np.ndarray:
    """keypoints: (T, 17, 3) gồm x, y (pixel), độ tin cậy. Trả mảng cùng hình dạng:
    - keypoint có độ tin cậy dưới ngưỡng được nội suy tuyến tính theo thời gian (giữ nguyên độ tin cậy gốc);
    - tịnh tiến về TRUNG BÌNH vị trí hông của cả cửa sổ (không phải từng khung, để giữ chuyển động đi lên/xuống);
    - chia cho độ dài thân (vai-hông) trung vị của cả cửa sổ, nên không phụ thuộc khoảng cách tới camera.
    Nếu không đo được thân (ví dụ mất hết vai/hông) thì chia cho kích thước lớn nhất của cả cửa sổ."""
    kp = np.asarray(keypoints, dtype=np.float32)
    t = kp.shape[0]
    xy, conf = kp[..., :2].copy(), kp[..., 2]
    xy[conf < conf_threshold] = np.nan
    steps = np.arange(t)
    for v in range(xy.shape[1]):
        ok = ~np.isnan(xy[:, v, 0])
        if 0 < ok.sum() < t:
            for c in range(2):
                xy[:, v, c] = np.interp(steps, steps[ok], xy[ok, v, c])
    hips = _nanmean(xy[:, HIPS, :], axis=1)
    shoulders = _nanmean(xy[:, SHOULDERS, :], axis=1)
    center = _nanmean(hips, axis=0)
    if np.isnan(center).any():
        center = np.zeros(2, dtype=np.float32)
    torso = _nanmedian(np.linalg.norm(shoulders - hips, axis=1))
    if not np.isfinite(torso) or torso < 1e-3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            extent = np.nanmax(xy.reshape(-1, 2), axis=0) - np.nanmin(xy.reshape(-1, 2), axis=0) if np.isfinite(xy).any() else np.zeros(2)
        torso = float(np.nanmax(extent)) if np.isfinite(extent).any() and np.nanmax(extent) > 1e-3 else 1.0
    out = np.empty_like(kp)
    out[..., :2] = np.nan_to_num((xy - center) / torso, nan=0.0)
    out[..., 2] = conf
    return out
