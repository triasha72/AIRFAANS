"""Typed mesh and graph representations used across AIRFAANS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlowCase:
    """One aerodynamic simulation sampled at mesh nodes."""

    case_id: str
    points: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    surface_mask: np.ndarray
    surface_normals: np.ndarray
    reynolds: float
    angle_of_attack_deg: float

    def validate(self) -> None:
        node_count = len(self.points)
        if self.points.shape != (node_count, 2):
            raise ValueError("points must have shape [nodes, 2]")
        if self.features.shape[0] != node_count or self.targets.shape != (node_count, 4):
            raise ValueError("features and four field targets must align with points")
        if self.surface_mask.shape != (node_count,):
            raise ValueError("surface_mask must have one value per node")
        if self.surface_normals.shape != (node_count, 2):
            raise ValueError("surface_normals must have shape [nodes, 2]")
        arrays = (self.points, self.features, self.targets, self.surface_normals)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("case contains non-finite values")


@dataclass(frozen=True)
class GraphCase:
    """Flow case with directed graph edges and relative edge features."""

    flow: FlowCase
    edge_index: np.ndarray
    edge_features: np.ndarray

    def validate(self) -> None:
        self.flow.validate()
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if self.edge_features.shape != (self.edge_index.shape[1], 3):
            raise ValueError("edge features must contain dx, dy and distance")
        if self.edge_index.size and (
            self.edge_index.min() < 0 or self.edge_index.max() >= len(self.flow.points)
        ):
            raise ValueError("edge index is outside the node range")
