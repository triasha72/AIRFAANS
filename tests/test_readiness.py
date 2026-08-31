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
