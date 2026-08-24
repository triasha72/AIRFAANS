"""Engineering quantities recovered from nodal flow predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airfaans.data import FlowCase


@dataclass(frozen=True)
class ForceCoefficients:
    lift: float
    drag: float


def pressure_coefficient(pressure: np.ndarray, density: float = 1.0, speed: float = 1.0):
    dynamic_pressure = 0.5 * density * speed**2
    if dynamic_pressure <= 0:
        raise ValueError("dynamic pressure must be positive")
    return np.asarray(pressure) / dynamic_pressure


def integrate_pressure_forces(
    case: FlowCase,
    predicted_fields: np.ndarray,
    density: float = 1.0,
    speed: float = 1.0,
    chord: float = 1.0,
) -> ForceCoefficients:
    """Integrate surface pressure around an ordered 2-D boundary approximation."""
    case.validate()
    fields = np.asarray(predicted_fields)
    if fields.shape != case.targets.shape:
        raise ValueError("predicted_fields must align with the case targets")
    indices = np.flatnonzero(case.surface_mask)
    if len(indices) < 3:
        raise ValueError("at least three surface nodes are required")
    points = case.points[indices]
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    points = points[order]
    normals = case.surface_normals[indices][order]
    cp = pressure_coefficient(fields[indices, 2][order], density, speed)
    next_points = np.roll(points, -1, axis=0)
    segment_length = np.linalg.norm(next_points - points, axis=1)
    force = -np.sum(cp[:, None] * normals * segment_length[:, None], axis=0) / chord
    alpha = np.deg2rad(case.angle_of_attack_deg)
    drag_direction = np.array([np.cos(alpha), np.sin(alpha)])
    lift_direction = np.array([-np.sin(alpha), np.cos(alpha)])
    return ForceCoefficients(
        lift=float(force @ lift_direction),
        drag=float(force @ drag_direction),
    )
