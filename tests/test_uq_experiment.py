import pytest

from airfaans.uq_experiment import compare_ood_uncertainty, summarize_uq_cases


def test_uq_case_summary_and_ood_comparison():
    cases = [
        {"mean_uncertainty": 1.0, "uncertainty_error_correlation": 0.2},
        {"mean_uncertainty": 2.0, "uncertainty_error_correlation": 0.4},
    ]
    assert summarize_uq_cases(cases) == {
        "mean_uncertainty": 1.5,
        "mean_uncertainty_error_correlation": pytest.approx(0.3),
    }
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b", "c"],
        "summary": {"mean_uncertainty": 1.5},
    }
    shifted = {
        "evaluation_task": "reynolds_ood",
        "checkpoint_sha256": ["a", "b", "c"],
        "summary": {"mean_uncertainty": 2.25},
    }
    comparison = compare_ood_uncertainty(base, shifted)
    assert comparison["ood_to_id_uncertainty_ratio"] == 1.5
    assert comparison["passed_predeclared_ratio"]
