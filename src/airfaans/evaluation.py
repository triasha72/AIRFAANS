"""Field, engineering, OOD and uncertainty metrics."""

from __future__ import annotations

import numpy as np

FIELD_NAMES = ("velocity_x", "velocity_y", "pressure", "turbulent_viscosity")


def field_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 4:
        raise ValueError("target and prediction must have shape [nodes, 4]")
    error = prediction - target
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    target_scale = np.sqrt(np.mean(target**2, axis=0))
    relative_l2 = rmse / np.maximum(target_scale, 1e-12)
    return {
        "node_count": len(target),
        "rmse": dict(zip(FIELD_NAMES, rmse.tolist(), strict=True)),
        "mae": dict(zip(FIELD_NAMES, mae.tolist(), strict=True)),
        "relative_l2": dict(zip(FIELD_NAMES, relative_l2.tolist(), strict=True)),
    }


def ensemble_summary(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 3 or predictions.shape[-1] != 4:
        raise ValueError("ensemble predictions must have shape [models, nodes, 4]")
    if len(predictions) < 2:
        raise ValueError("uncertainty requires at least two ensemble members")
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)


def uncertainty_error_correlation(
    target: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray
) -> float:
    error = np.linalg.norm(np.asarray(mean) - np.asarray(target), axis=1)
    uncertainty = np.linalg.norm(np.asarray(standard_deviation), axis=1)
    if np.std(error) == 0 or np.std(uncertainty) == 0:
        return 0.0
    return float(np.corrcoef(error, uncertainty)[0, 1])


def ood_uncertainty_ratio(id_uncertainty: np.ndarray, ood_uncertainty: np.ndarray) -> float:
    baseline = float(np.mean(id_uncertainty))
    if baseline <= 0:
        raise ValueError("ID uncertainty mean must be positive")
    return float(np.mean(ood_uncertainty) / baseline)
