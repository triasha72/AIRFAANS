"""Leakage-safe feature and target normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Normalization:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.feature_mean) / self.feature_scale).astype(np.float32)

    def transform_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.target_mean) / self.target_scale).astype(np.float32)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.target_scale + self.target_mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> Normalization:
        return cls(
            *(
                np.asarray(payload[name], dtype=np.float32)
                for name in ("feature_mean", "feature_scale", "target_mean", "target_scale")
            )
        )


class RunningMoments:
    """Numerically stable per-column moments without retaining all nodes."""

    def __init__(self, width: int) -> None:
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not len(values):
            raise ValueError("values must be a non-empty matrix with the configured width")
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    def finish(self, minimum_scale: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("at least two observations are required")
        scale = np.sqrt(self.m2 / self.count)
        return self.mean.astype(np.float32), np.maximum(scale, minimum_scale).astype(np.float32)
