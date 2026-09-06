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
    uq = next(
        (
            item
            for item in summaries
            if item.get("evidence_label") == "airfrans_ensemble_uq_summary"
        ),
        None,
    )
    active = next(
        (
            item
            for item in summaries
            if item.get("evidence_label") == "airfrans_active_learning_summary"
        ),
        None,
    )
    reynolds_ratio = None if uq is None else uq.get("reynolds_ood_to_id_uncertainty_ratio")
    aoa_ratio = None if uq is None else uq.get("aoa_ood_to_id_uncertainty_ratio")
    correlation = None if uq is None else uq.get("mean_uncertainty_error_correlation")
    active_gain = None if active is None else active.get("uncertainty_minus_random_error_reduction")
    checks = {
        "all_required_tasks": {"passed": not missing_tasks, "missing": missing_tasks},
        "three_models_on_interpolation": {"passed": model_count == 3, "value": model_count},
        "three_seeds_on_interpolation": {"passed": len(seeds) >= 3, "value": seeds},
        "ensemble_uncertainty_evidence": {"passed": uq is not None},
        "active_learning_evidence": {"passed": active is not None},
        "reynolds_ood_uncertainty_separation": {
            "value": reynolds_ratio,
            "minimum": 1.1,
            "passed": reynolds_ratio is not None and reynolds_ratio >= 1.1,
        },
        "aoa_ood_uncertainty_separation": {
            "value": aoa_ratio,
            "minimum": 1.1,
            "passed": aoa_ratio is not None and aoa_ratio >= 1.1,
        },
        "uncertainty_tracks_error": {
            "value": correlation,
            "minimum": 0.3,
            "passed": correlation is not None and correlation >= 0.3,
        },
        "active_learning_beats_random": {
            "value": active_gain,
            "minimum_exclusive": 0.0,
            "passed": active_gain is not None and active_gain > 0.0,
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
