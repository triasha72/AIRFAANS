"""Evidence completeness checks for scientific-model operational readiness."""

from __future__ import annotations

REQUIRED_TASKS = ("interpolation", "scarce", "reynolds_ood", "aoa_ood")


def assess_readiness(summaries: list[dict[str, object]]) -> dict[str, object]:
    official = [
        item
        for item in summaries
        if item.get("evidence_label") == "airfrans_official_three_seed_summary"
    ]
    completed_tasks = sorted({str(item.get("task")) for item in official})
    missing_tasks = [task for task in REQUIRED_TASKS if task not in completed_tasks]
    interpolation = next((item for item in official if item.get("task") == "interpolation"), None)
    model_count = len(interpolation.get("models", {})) if interpolation else 0
    seeds = interpolation.get("seeds", []) if interpolation else []
    checks = {
        "all_required_tasks": {"passed": not missing_tasks, "missing": missing_tasks},
        "three_models_on_interpolation": {"passed": model_count == 3, "value": model_count},
        "three_seeds_on_interpolation": {"passed": len(seeds) >= 3, "value": seeds},
        "ensemble_uncertainty_evidence": {
            "passed": any(
                item.get("evidence_label") == "airfrans_ensemble_uq_summary" for item in summaries
            )
        },
        "active_learning_evidence": {
            "passed": any(
                item.get("evidence_label") == "airfrans_active_learning_summary"
                for item in summaries
            )
        },
    }
    return {
        "schema_version": "1.0",
        "policy": "airfaans-operational-readiness-v1",
        "decision": "approved" if all(item["passed"] for item in checks.values()) else "rejected",
        "completed_tasks": completed_tasks,
        "checks": checks,
        "next_experiments": [
            "Run three matched seeds for scarce, Reynolds-OOD, and AoA-OOD tasks.",
            "Measure ensemble calibration, uncertainty-error correlation, "
            "and OOD/ID uncertainty ratio.",
            "Compare uncertainty acquisition with seeded random acquisition under equal compute.",
        ],
    }
