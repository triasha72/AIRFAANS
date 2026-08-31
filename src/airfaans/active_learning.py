"""Simulation-case selection for active-learning experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def select_high_uncertainty(
    case_ids: Sequence[str], uncertainty: Sequence[float], count: int
) -> list[str]:
    if len(case_ids) != len(uncertainty):
        raise ValueError("case IDs and uncertainty scores must align")
    if not 0 < count <= len(case_ids):
        raise ValueError("count must select at least one available case")
    ranked = sorted(
        zip(case_ids, np.asarray(uncertainty, dtype=float), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return [case_id for case_id, _ in ranked[:count]]


def select_random(case_ids: Sequence[str], count: int, seed: int) -> list[str]:
    if not 0 < count <= len(case_ids):
        raise ValueError("count must select at least one available case")
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(case_ids), size=count, replace=False)
    return [case_ids[index] for index in sorted(indices)]


def build_acquisition_round(
    uq_report: dict[str, object], count: int, seed: int
) -> dict[str, object]:
    cases = uq_report["per_case"]
    case_ids = [case["case_id"] for case in cases]
    uncertainty = [case["mean_uncertainty"] for case in cases]
    return {
        "schema_version": "1.0",
        "evidence_label": "airfrans_active_learning_acquisition",
        "source_checkpoints": uq_report["checkpoint_sha256"],
        "count": count,
        "uncertainty_selected": select_high_uncertainty(case_ids, uncertainty, count),
        "random_selected": select_random(case_ids, count, seed),
        "random_seed": seed,
        "fairness_contract": "retrain both arms from scratch with equal model and compute budgets",
    }


def write_acquisition_round(report_path: Path, output_path: Path, count: int, seed: int):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    result = build_acquisition_round(report, count, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
