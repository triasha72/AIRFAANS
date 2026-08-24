"""AirfRANS/OpenFOAM VTK ingestion boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from airfaans.data import FlowCase


def load_vtk_case(
    path: Path,
    case_id: str,
    reynolds: float,
    angle_of_attack_deg: float,
) -> FlowCase:
    """Read a processed `.vtu`/`.vtp` case with AirfRANS-style fields.

    Expected point arrays are `Velocity`, `Pressure`, `TurbulentViscosity`,
    `Distance`, `Normals`, and optionally `Surface`. Dataset-specific aliases
    should be normalized during the preprocessing step, not guessed here.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'airfrans' extra for VTK ingestion.") from exc
    mesh = pv.read(path)
    points = np.asarray(mesh.points[:, :2], dtype=np.float32)

    def required(name: str) -> np.ndarray:
        if name not in mesh.point_data:
            raise ValueError(f"{path} is missing required point array: {name}")
        return np.asarray(mesh.point_data[name])

    velocity = required("Velocity")[:, :2]
    pressure = required("Pressure").reshape(-1)
    turbulent_viscosity = required("TurbulentViscosity").reshape(-1)
    distance = required("Distance").reshape(-1)
    normals = required("Normals")[:, :2]
    surface = (
        np.asarray(mesh.point_data["Surface"]).reshape(-1).astype(bool)
        if "Surface" in mesh.point_data
        else np.isclose(distance, 0.0)
    )
    alpha = np.deg2rad(angle_of_attack_deg)
    features = np.column_stack(
        (
            points,
            np.full(len(points), np.cos(alpha)),
            np.full(len(points), np.sin(alpha)),
            distance,
            normals,
            np.full(len(points), np.log10(reynolds)),
            np.full(len(points), alpha),
        )
    )
    case = FlowCase(
        case_id=case_id,
        points=points,
        features=features.astype(np.float32),
        targets=np.column_stack((velocity, pressure, turbulent_viscosity)).astype(np.float32),
        surface_mask=surface,
        surface_normals=normals.astype(np.float32),
        reynolds=reynolds,
        angle_of_attack_deg=angle_of_attack_deg,
    )
    case.validate()
    return case


def save_case(path: Path, case: FlowCase) -> None:
    case.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case_id=case.case_id,
        points=case.points,
        features=case.features,
        targets=case.targets,
        surface_mask=case.surface_mask,
        surface_normals=case.surface_normals,
        reynolds=case.reynolds,
        angle_of_attack_deg=case.angle_of_attack_deg,
    )


def load_case(path: Path) -> FlowCase:
    payload = np.load(path, allow_pickle=False)
    case = FlowCase(
        case_id=str(payload["case_id"]),
        points=payload["points"],
        features=payload["features"],
        targets=payload["targets"],
        surface_mask=payload["surface_mask"],
        surface_normals=payload["surface_normals"],
        reynolds=float(payload["reynolds"]),
        angle_of_attack_deg=float(payload["angle_of_attack_deg"]),
    )
    case.validate()
    return case
