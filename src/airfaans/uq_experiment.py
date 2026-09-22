"""Executable deep-ensemble and OOD uncertainty experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EnsembleManifest:
    """Identity boundary for a frozen set of ensemble checkpoints."""

    model: str
    training_task: str
    seeds: tuple[int, ...]
    checkpoint_paths: tuple[str, ...]
    checkpoint_sha256: tuple[str, ...]


@dataclass(frozen=True)
class EnsembleAudit:
    expected_cases: int
    saved_cases: int
    next_missing_index: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ensemble_manifest(checkpoint_paths: list[Path]) -> EnsembleManifest:
    """Load checkpoint metadata and reject an ensemble with mixed identity."""
    import torch

    if len(checkpoint_paths) < 2:
        raise ValueError("deep-ensemble evaluation requires at least two checkpoints")
    identities: list[tuple[str, str, int]] = []
    hashes: list[str] = []
    resolved_paths: list[str] = []
    for path in checkpoint_paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"checkpoint has no serialized config: {path}")
        model, task, seed = config.get("model"), config.get("task"), config.get("seed")
        if not isinstance(model, str) or not isinstance(task, str) or not isinstance(seed, int):
            raise ValueError(f"checkpoint config is incomplete: {path}")
        identities.append((model, task, seed))
        hashes.append(_sha256(path))
        resolved_paths.append(str(path.resolve()))
    model_tasks = {(model, task) for model, task, _ in identities}
    if len(model_tasks) != 1:
        raise ValueError("ensemble members must share model and training task")
    seeds = tuple(seed for _, _, seed in identities)
    if len(set(seeds)) != len(seeds):
        raise ValueError("ensemble members must use distinct seeds")
    model, task = next(iter(model_tasks))
    return EnsembleManifest(
        model=model,
        training_task=task,
        seeds=seeds,
        checkpoint_paths=tuple(resolved_paths),
        checkpoint_sha256=tuple(hashes),
    )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Make a case record visible only after its JSON body is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_ensemble_records(
    records_dir: Path,
    manifest: EnsembleManifest,
    evaluation_task: str,
    expected_cases: int,
) -> EnsembleAudit:
    """Fail closed on incomplete, duplicate, or mixed-provenance case records."""
    if expected_cases <= 0:
        raise ValueError("expected_cases must be positive")
    indexes: set[int] = set()
    for record_path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON record: {record_path}") from error
        provenance = (
            record.get("evaluation_task"),
            record.get("model"),
            record.get("training_task"),
            tuple(record.get("seeds", [])),
            tuple(record.get("checkpoint_sha256", [])),
        )
        expected = (
            evaluation_task,
            manifest.model,
            manifest.training_task,
            manifest.seeds,
            manifest.checkpoint_sha256,
        )
        if provenance != expected:
            raise ValueError(
                f"mixed-provenance record rejected: {record_path.name}; "
                f"expected={expected!r}, found={provenance!r}"
            )
        index = record.get("official_test_index")
        if not isinstance(index, int) or not 0 <= index < expected_cases:
            raise ValueError(f"invalid official_test_index in {record_path.name}: {index!r}")
        if index in indexes:
            raise ValueError(f"duplicate official_test_index rejected: {index}")
        indexes.add(index)
    return EnsembleAudit(
        expected_cases=expected_cases,
        saved_cases=len(indexes),
        next_missing_index=next((index for index in range(expected_cases) if index not in indexes), None),
    )


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
