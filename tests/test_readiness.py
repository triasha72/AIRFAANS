import json
from pathlib import Path

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
