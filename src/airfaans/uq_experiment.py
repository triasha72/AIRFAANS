"""Executable deep-ensemble and OOD uncertainty experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from airfaans.airfrans import load_case
from airfaans.evaluation import ensemble_summary, field_metrics, uncertainty_error_correlation
from airfaans.experiment import (
    ExperimentConfig,
    _case_tensors,
    _forward,
    build_model,
    case_directory,
    official_split,
    sample_indices,
)
from airfaans.normalization import Normalization


def summarize_uq_cases(per_case: list[dict[str, object]]) -> dict[str, float]:
    if not per_case:
        raise ValueError("at least one case is required")
    return {
        "mean_uncertainty": float(np.mean([case["mean_uncertainty"] for case in per_case])),
        "mean_uncertainty_error_correlation": float(
            np.mean([case["uncertainty_error_correlation"] for case in per_case])
        ),
    }


def compare_ood_uncertainty(id_report: dict[str, object], ood_report: dict[str, object]):
    if id_report["checkpoint_sha256"] != ood_report["checkpoint_sha256"]:
        raise ValueError("ID and OOD reports must use the same ensemble checkpoints")
    baseline = float(id_report["summary"]["mean_uncertainty"])
    if baseline <= 0:
        raise ValueError("ID uncertainty must be positive")
    ratio = float(ood_report["summary"]["mean_uncertainty"]) / baseline
    return {
        "schema_version": "1.0",
        "evidence_label": "airfrans_ood_uncertainty_comparison",
        "id_task": id_report["evaluation_task"],
        "ood_task": ood_report["evaluation_task"],
        "checkpoint_sha256": id_report["checkpoint_sha256"],
        "ood_to_id_uncertainty_ratio": ratio,
        "passed_predeclared_ratio": ratio > 1.0,
    }


def evaluate_ensemble(
    dataset_root: Path,
    manifest_path: Path,
    checkpoint_paths: list[Path],
    evaluation_task: str,
    output_path: Path,
    max_cases: int | None = None,
) -> dict[str, object]:
    import torch

    if len(checkpoint_paths) < 2:
        raise ValueError("deep-ensemble evaluation requires at least two checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = []
    configs = []
    hashes = []
    for path in checkpoint_paths:
        payload = torch.load(path, map_location=device, weights_only=True)
        config = ExperimentConfig(**payload["config"])
        normalization = Normalization.from_dict(payload["normalization"])
        model = build_model(config, len(normalization.feature_mean)).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        members.append((model, normalization))
        configs.append(config)
        hashes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    identity = {(config.model, config.task) for config in configs}
    if len(identity) != 1 or len({config.seed for config in configs}) != len(configs):
        raise ValueError("ensemble members must share model/task and use distinct seeds")
    _, _, case_ids = official_split(manifest_path, evaluation_task, configs[0].validation_cases)
    if max_cases:
        case_ids = case_ids[:max_cases]
    per_case = []
    with torch.inference_mode():
        for position, case_id in enumerate(case_ids):
            case = load_case(case_directory(dataset_root, case_id))
            indices = sample_indices(case, configs[0].nodes_per_case, 900_000 + position)
            predictions = []
            for (model, normalization), config in zip(members, configs, strict=True):
                x, _, edges, edge_features = _case_tensors(
                    case, normalization, indices, device, config.model
                )
                normalized = _forward(model, config.model, x, edges, edge_features)
                predictions.append(normalization.inverse_targets(normalized.cpu().numpy()))
            mean, standard_deviation = ensemble_summary(np.asarray(predictions))
            per_case.append(
                {
                    "case_id": case_id,
                    "field_metrics": field_metrics(case.targets[indices], mean),
                    "mean_uncertainty": float(np.mean(np.linalg.norm(standard_deviation, axis=1))),
                    "uncertainty_error_correlation": uncertainty_error_correlation(
                        case.targets[indices], mean, standard_deviation
                    ),
                }
            )
    result = {
        "schema_version": "1.0",
        "evidence_label": "airfrans_ensemble_uq_summary",
        "training_task": configs[0].task,
        "evaluation_task": evaluation_task,
        "model": configs[0].model,
        "seeds": [config.seed for config in configs],
        "checkpoint_sha256": hashes,
        "case_count": len(per_case),
        "bounded": max_cases is not None,
        "summary": summarize_uq_cases(per_case),
        "per_case": per_case,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
