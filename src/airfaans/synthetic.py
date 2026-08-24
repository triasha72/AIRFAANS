"""Small analytic aerodynamic fixtures for tests and first-run demos.

These fields are not RANS CFD and must never be reported as AirfRANS results.
"""

from __future__ import annotations

import numpy as np

from airfaans.data import FlowCase


def make_case(
    case_id: str = "synthetic-naca-0012",
    reynolds: float = 1.0e6,
    angle_of_attack_deg: float = 4.0,
    radial_count: int = 12,
    angular_count: int = 32,
) -> FlowCase:
    """Create a deterministic cylinder-like flow fixture around a thin body."""
    radii = np.linspace(0.55, 2.5, radial_count)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_count, endpoint=False)
    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    points = np.column_stack((rr.ravel() * np.cos(tt).ravel(), rr.ravel() * np.sin(tt).ravel()))
    surface_mask = np.repeat(np.arange(radial_count) == 0, angular_count)
    normals = np.column_stack((np.cos(tt).ravel(), np.sin(tt).ravel()))
    alpha = np.deg2rad(angle_of_attack_deg)
    freestream = np.array([np.cos(alpha), np.sin(alpha)])
    radius_sq = np.maximum(np.sum(points**2, axis=1), 0.55**2)
    disturbance = 0.55**2 / radius_sq
    ux = freestream[0] * (1.0 - disturbance * np.cos(2.0 * tt.ravel()))
    uy = freestream[1] - freestream[0] * disturbance * np.sin(2.0 * tt.ravel())
    speed_sq = ux**2 + uy**2
    pressure = 0.5 * (1.0 - speed_sq)
    turbulent_viscosity = np.maximum(0.0, 0.015 * np.exp(-(rr.ravel() - 0.55)))
    distance = rr.ravel() - 0.55
    features = np.column_stack(
        (
            points,
            np.full(len(points), freestream[0]),
            np.full(len(points), freestream[1]),
            distance,
            normals,
            np.full(len(points), np.log10(reynolds)),
            np.full(len(points), alpha),
        )
    )
    targets = np.column_stack((ux, uy, pressure, turbulent_viscosity))
    case = FlowCase(
        case_id=case_id,
        points=points.astype(np.float32),
        features=features.astype(np.float32),
        targets=targets.astype(np.float32),
        surface_mask=surface_mask,
        surface_normals=normals.astype(np.float32),
        reynolds=reynolds,
        angle_of_attack_deg=angle_of_attack_deg,
    )
    case.validate()
    return case
