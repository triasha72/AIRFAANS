"""Adapter for the official processed AirfRANS VTK dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from airfaans.data import FlowCase, GraphCase
from airfaans.graph import from_cells


@dataclass(frozen=True)
class AirfRANSCondition:
    inlet_speed: float
    angle_of_attack_deg: float
    reynolds: float


def parse_condition(case_id: str, temperature_kelvin: float = 298.15) -> AirfRANSCondition:
    """Parse freestream speed/AoA and reproduce the official Reynolds convention."""
    parts = case_id.split("_")
    if len(parts) < 4 or parts[0] != "airFoil2D":
        raise ValueError(f"Unrecognized AirfRANS case name: {case_id}")
    inlet_speed = float(parts[2])
    angle = float(parts[3])
    temperature = float(temperature_kelvin)
    viscosity = (
        -3.400747e-6
        + 3.452139e-8 * temperature
        + 1.00881778e-10 * temperature**2
        - 1.363528e-14 * temperature**3
    )
    return AirfRANSCondition(inlet_speed, angle, inlet_speed / viscosity)


def _nearest_normals(
    surface_points: np.ndarray, airfoil_points: np.ndarray, airfoil_normals: np.ndarray
) -> np.ndarray:
    """Match airfoil-patch normals to coincident internal surface nodes."""
    result = np.empty((len(surface_points), 2), dtype=np.float32)
    for start in range(0, len(surface_points), 256):
        query = surface_points[start : start + 256]
        distance_sq = np.sum((query[:, None, :] - airfoil_points[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distance_sq, axis=1)
        error = np.sqrt(distance_sq[np.arange(len(query)), nearest])
        if np.max(error, initial=0.0) > 1e-5:
            raise ValueError(f"Surface/airfoil point mismatch: maximum distance {error.max():.3e}")
        result[start : start + len(query)] = airfoil_normals[nearest]
    return result


def _cells(mesh) -> np.ndarray:
    """Decode PyVista's `[count, ids...]` connectivity into padded rows."""
    encoded = np.asarray(mesh.cells, dtype=np.int64)
    rows: list[np.ndarray] = []
    cursor = 0
    while cursor < len(encoded):
        count = int(encoded[cursor])
        rows.append(encoded[cursor + 1 : cursor + 1 + count])
        cursor += count + 1
    if cursor != len(encoded) or not rows:
        raise ValueError("Invalid or empty VTK cell connectivity")
    width = max(map(len, rows))
    padded = np.full((len(rows), width), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = row
    return padded


def load_case(case_directory: Path) -> FlowCase:
    """Load one official processed AirfRANS simulation."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'airfrans' extra.") from exc
    case_directory = Path(case_directory)
    case_id = case_directory.name
    condition = parse_condition(case_id)
    internal = pv.read(case_directory / f"{case_id}_internal.vtu")
    airfoil = pv.read(case_directory / f"{case_id}_aerofoil.vtp")
    required_internal = {"U", "p", "nut", "implicit_distance"}
    required_airfoil = {"Normals"}
    if missing := required_internal - set(internal.point_data):
        raise ValueError(f"{case_id} internal mesh missing fields: {sorted(missing)}")
    if missing := required_airfoil - set(airfoil.point_data):
        raise ValueError(f"{case_id} airfoil mesh missing fields: {sorted(missing)}")

    points = np.asarray(internal.points[:, :2], dtype=np.float32)
    velocity = np.asarray(internal.point_data["U"][:, :2], dtype=np.float32)
    pressure = np.asarray(internal.point_data["p"], dtype=np.float32).reshape(-1)
    turbulent_viscosity = np.asarray(internal.point_data["nut"], dtype=np.float32).reshape(-1)
    distance = -np.asarray(internal.point_data["implicit_distance"], dtype=np.float32).reshape(-1)
    surface = velocity[:, 0] == 0.0
    normals = np.zeros((len(points), 2), dtype=np.float32)
    normals[surface] = _nearest_normals(
        points[surface],
        np.asarray(airfoil.points[:, :2], dtype=np.float32),
        np.asarray(airfoil.point_data["Normals"][:, :2], dtype=np.float32),
    )
    alpha = np.deg2rad(condition.angle_of_attack_deg)
    inlet = condition.inlet_speed * np.array([np.cos(alpha), np.sin(alpha)])
    features = np.column_stack(
        (
            points,
            np.tile(inlet, (len(points), 1)),
            distance,
            normals,
            np.full(len(points), np.log10(condition.reynolds)),
            np.full(len(points), alpha),
        )
    )
    case = FlowCase(
        case_id=case_id,
        points=points,
        features=features.astype(np.float32),
        targets=np.column_stack((velocity, pressure, turbulent_viscosity)).astype(np.float32),
        surface_mask=surface,
        surface_normals=normals,
        reynolds=condition.reynolds,
        angle_of_attack_deg=condition.angle_of_attack_deg,
    )
    case.validate()
    return case


def load_graph(case_directory: Path) -> GraphCase:
    """Load a case and preserve official VTK mesh-cell connectivity."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'airfrans' extra.") from exc
    case_directory = Path(case_directory)
    case_id = case_directory.name
    mesh = pv.read(case_directory / f"{case_id}_internal.vtu")
    return from_cells(load_case(case_directory), _cells(mesh))
