import numpy as np
import pytest

from airfaans.evaluation import (
    ensemble_summary,
    field_metrics,
    ood_uncertainty_ratio,
    uncertainty_error_correlation,
)
from airfaans.physics import _nearest_values, integrate_pressure_forces
from airfaans.synthetic import make_case


def test_identity_field_metrics_are_zero():
    target = make_case(radial_count=3, angular_count=8).targets
    metrics = field_metrics(target, target.copy())
    assert all(value == pytest.approx(0.0) for value in metrics["rmse"].values())


def test_ensemble_and_ood_metrics():
    target = np.zeros((5, 4))
    predictions = np.stack((target + 1.0, target + 3.0))
    mean, standard_deviation = ensemble_summary(predictions)
    assert np.all(mean == 2.0)
    assert np.all(standard_deviation > 0)
    assert uncertainty_error_correlation(target, mean, standard_deviation) == 0.0
    assert ood_uncertainty_ratio(np.ones(5), np.ones(5) * 2) == pytest.approx(2.0)


def test_pressure_force_integration_returns_finite_coefficients():
    case = make_case()
    coefficients = integrate_pressure_forces(case, case.targets)
    assert np.isfinite(coefficients.lift)
    assert np.isfinite(coefficients.drag)


def test_nearest_surface_mapping_is_order_independent():
    source = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    values = np.array([[10.0], [20.0], [30.0]])
    query = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(_nearest_values(query, source, values).ravel(), [30, 10, 20])
