import json
from pathlib import Path

import pytest

from airfaans.readiness import assess_readiness


def test_readiness_reports_missing_ood_and_uq_evidence():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (root / "artifacts/evaluation/interpolation_three_seed_summary.json").read_text()
    )
    result = assess_readiness([summary])
    assert result["decision"] == "rejected"
    assert result["checks"]["three_models_on_interpolation"]["passed"]
    assert result["checks"]["three_seeds_on_interpolation"]["passed"]
    assert result["checks"]["all_required_tasks"]["missing"] == [
        "scarce",
        "reynolds_ood",
        "aoa_ood",
    ]
    assert not result["checks"]["uncertainty_tracks_error"]["passed"]


def test_readiness_requires_useful_uq_not_just_a_report_file():
    result = assess_readiness(
        [
            {
                "evidence_label": "airfrans_ensemble_uq_summary",
                "reynolds_ood_to_id_uncertainty_ratio": 1.2,
                "aoa_ood_to_id_uncertainty_ratio": 1.05,
                "mean_uncertainty_error_correlation": 0.4,
            },
            {
                "evidence_label": "airfrans_active_learning_summary",
                "uncertainty_minus_random_error_reduction": -0.01,
            },
        ]
    )
    assert result["checks"]["ensemble_uncertainty_evidence"]["passed"]
    assert result["checks"]["uncertainty_tracks_error"]["passed"]
    assert not result["checks"]["aoa_ood_uncertainty_separation"]["passed"]
    assert not result["checks"]["active_learning_beats_random"]["passed"]


def test_readiness_rejects_repeated_seeds():
    result = assess_readiness(
        [
            {
                "evidence_label": "airfrans_official_three_seed_summary",
                "task": "interpolation",
                "models": {"a": {}, "b": {}, "c": {}},
                "seeds": [17, 17, 17],
            }
        ]
    )
    assert not result["checks"]["three_seeds_on_interpolation"]["passed"]


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), "1.5", True])
def test_readiness_fails_closed_on_invalid_uq_metrics(invalid):
    result = assess_readiness(
        [
            {
                "evidence_label": "airfrans_ensemble_uq_summary",
                "reynolds_ood_to_id_uncertainty_ratio": invalid,
                "aoa_ood_to_id_uncertainty_ratio": invalid,
                "mean_uncertainty_error_correlation": invalid,
            }
        ]
    )
    assert not result["checks"]["reynolds_ood_uncertainty_separation"]["passed"]
    assert not result["checks"]["uncertainty_tracks_error"]["passed"]


def test_readiness_consumes_emitted_ensemble_and_comparison_reports():
    reports = [
        {
            "evidence_label": "airfrans_ensemble_uq_summary",
            "bounded": False,
            "evaluation_task": "interpolation",
            "checkpoint_sha256": ["a", "b"],
            "summary": {"mean_uncertainty_error_correlation": 0.5},
        }
    ]
    for task, ratio in (("reynolds_ood", 1.2), ("aoa_ood", 1.3)):
        reports.append(
            {
                "evidence_label": "airfrans_ood_uncertainty_comparison",
                "id_task": "interpolation",
                "ood_task": task,
                "checkpoint_sha256": ["a", "b"],
                "ood_to_id_uncertainty_ratio": ratio,
            }
        )
    result = assess_readiness(reports)
    assert result["checks"]["reynolds_ood_uncertainty_separation"]["passed"]
    assert result["checks"]["aoa_ood_uncertainty_separation"]["passed"]
    assert result["checks"]["uncertainty_tracks_error"]["passed"]
    reports[-1]["checkpoint_sha256"] = ["other", "ensemble"]
    assert not assess_readiness(reports)["checks"]["aoa_ood_uncertainty_separation"]["passed"]


def test_readiness_ignores_bounded_uq_reports():
    result = assess_readiness(
        [
            {
                "evidence_label": "airfrans_ensemble_uq_summary",
                "bounded": True,
                "reynolds_ood_to_id_uncertainty_ratio": 1.5,
                "aoa_ood_to_id_uncertainty_ratio": 1.5,
                "mean_uncertainty_error_correlation": 0.5,
            }
        ]
    )
    assert not result["checks"]["ensemble_uncertainty_evidence"]["passed"]


def test_readiness_matches_reports_independently_of_file_order():
    unrelated = {
        "evidence_label": "airfrans_ensemble_uq_summary",
        "bounded": False,
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["other", "model"],
        "summary": {"mean_uncertainty_error_correlation": 0.1},
    }
    matched = {
        **unrelated,
        "checkpoint_sha256": ["a", "b"],
        "summary": {"mean_uncertainty_error_correlation": 0.5},
    }
    comparisons = [
        {
            "evidence_label": "airfrans_ood_uncertainty_comparison",
            "id_task": "interpolation",
            "ood_task": task,
            "checkpoint_sha256": ["a", "b"],
            "ood_to_id_uncertainty_ratio": 1.2,
        }
        for task in ("reynolds_ood", "aoa_ood")
    ]
    for reports in ([unrelated, matched, *comparisons], [matched, unrelated, *comparisons]):
        checks = assess_readiness(reports)["checks"]
        assert checks["reynolds_ood_uncertainty_separation"]["passed"]
        assert checks["aoa_ood_uncertainty_separation"]["passed"]
        assert checks["uncertainty_tracks_error"]["passed"]
