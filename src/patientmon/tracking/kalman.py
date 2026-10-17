"""Bộ lọc Kalman cho hộp bao, trạng thái (cx, cy, tỉ lệ w/h, h) và vận tốc tương ứng."""
from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(self):
        ndim = 4
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = 1.0
        self._update = np.eye(ndim, 2 * ndim)
        self._wp, self._wv = 1.0 / 20, 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-2, self._wp * h, self._wv * h, self._wv * h, 1e-5, self._wv * h]
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray):
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-1, self._wp * h]
        return self._update @ mean, self._update @ cov @ self._update.T + np.diag(np.square(std))

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        proj_mean, proj_cov = self.project(mean, cov)
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        new_mean = mean + gain @ (measurement - proj_mean)
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov
