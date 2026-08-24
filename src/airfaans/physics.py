"""Engineering quantities recovered from nodal flow predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from airfaans.data import FlowCase


@dataclass(frozen=True)
class ForceCoefficients:
    lift: float
    drag: float


@dataclass(frozen=True)
class AirfRANSForceCoefficients:
    drag: float
    pressure_drag: float
    viscous_drag: float
    lift: float
    pressure_lift: float
    viscous_lift: float


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


def _nearest_values(query: np.ndarray, source: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map coincident surface arrays without relying on source ordering."""
    result = np.empty((len(query), *values.shape[1:]), dtype=np.float64)
    for start in range(0, len(query), 256):
        points = query[start : start + 256]
        distance_sq = np.sum((points[:, None] - source[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distance_sq, axis=1)
        error = np.sqrt(distance_sq[np.arange(len(points)), nearest])
        if np.max(error, initial=0.0) > 1e-5:
            raise ValueError(f"surface mapping mismatch: maximum distance {error.max():.3e}")
        result[start : start + len(points)] = values[nearest]
    return result


def integrate_airfrans_forces(
    case_directory,
    predicted_fields: np.ndarray | None = None,
    temperature_kelvin: float = 298.15,
) -> AirfRANSForceCoefficients:
    """Reproduce the official AirfRANS pressure and viscous-force convention.

    ``predicted_fields`` must cover every internal-mesh node in ``[U_x,U_y,p,nut]``
    order. Turbulent viscosity is not part of the official wall-stress expression;
    the molecular kinematic viscosity and deviatoric velocity gradient are used.
    """
    from pathlib import Path

    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'airfrans' extra.") from exc

    path = Path(case_directory)
    case_id = path.name
    internal = pv.read(path / f"{case_id}_internal.vtu")
    airfoil = pv.read(path / f"{case_id}_aerofoil.vtp").compute_cell_sizes(area=False, volume=False)
    reference = np.column_stack(
        (internal.point_data["U"][:, :2], internal.point_data["p"], internal.point_data["nut"])
    )
    fields = reference if predicted_fields is None else np.asarray(predicted_fields, dtype=float)
    if fields.shape != reference.shape or not np.isfinite(fields).all():
        raise ValueError("predicted_fields must be finite and cover every internal mesh node")
    surface = np.asarray(internal.point_data["U"][:, 0] == 0.0)
    internal_points = np.asarray(internal.points[surface, :2])
    airfoil_points = np.asarray(airfoil.points[:, :2])
    normals_internal = _nearest_values(
        internal_points,
        airfoil_points,
        np.asarray(airfoil.point_data["Normals"][:, :2]),
    )
    velocity_3d = np.column_stack((fields[:, :2], np.zeros(len(fields))))
    derivative_mesh = internal.copy()
    derivative_mesh.point_data["predicted_U"] = velocity_3d
    jacobian = np.asarray(
        derivative_mesh.compute_derivative(scalars="predicted_U", gradient="jacobian")
        .point_data["jacobian"]
        .reshape(-1, 3, 3)[surface, :2, :2]
    )
    strain = 0.5 * (jacobian + jacobian.transpose(0, 2, 1))
    strain -= strain.trace(axis1=-2, axis2=-1)[:, None, None] * np.eye(2)[None] / 3.0
    temperature = float(temperature_kelvin)
    nu = (
        -3.400747e-6
        + 3.452139e-8 * temperature
        + 1.00881778e-10 * temperature**2
        - 1.363528e-14 * temperature**3
    )
    wall_shear_internal = -(2.0 * nu * strain * normals_internal[:, None, :]).sum(axis=2)
    wall_shear = _nearest_values(airfoil_points, internal_points, wall_shear_internal)
    pressure = _nearest_values(airfoil_points, internal_points, fields[surface, 2, None])[:, 0]
    surface_mesh = airfoil.copy()
    surface_mesh.point_data["wallShearStress"] = wall_shear
    surface_mesh.point_data["predicted_p"] = pressure
    cells = surface_mesh.point_data_to_cell_data(pass_point_data=False)
    lengths = np.asarray(cells.cell_data["Length"])
    cell_normals = np.asarray(cells.cell_data["Normals"][:, :2])
    pressure_force = np.sum(
        np.asarray(cells.cell_data["predicted_p"])[:, None] * cell_normals * lengths[:, None],
        axis=0,
    )
    viscous_force = np.sum(
        np.asarray(cells.cell_data["wallShearStress"]) * lengths[:, None], axis=0
    )
    inlet_speed = float(case_id.split("_")[2])
    alpha = np.deg2rad(float(case_id.split("_")[3]))
    basis = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    denominator = 0.5 * inlet_speed**2
    pressure_coefficients = basis @ pressure_force / denominator
    viscous_coefficients = basis @ viscous_force / denominator
    total = pressure_coefficients + viscous_coefficients
    return AirfRANSForceCoefficients(
        drag=float(total[0]),
        pressure_drag=float(pressure_coefficients[0]),
        viscous_drag=float(viscous_coefficients[0]),
        lift=float(total[1]),
        pressure_lift=float(pressure_coefficients[1]),
        viscous_lift=float(viscous_coefficients[1]),
    )
