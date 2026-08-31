"""Fail-closed release evidence validation for checkpoint-backed inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class ReleaseValidationError(ValueError):
    """Raised when deployment evidence does not satisfy the release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(payload: dict[str, object], name: str, expected_type):
    value = payload.get(name)
    if not isinstance(value, expected_type):
        raise ReleaseValidationError(f"release manifest requires {name}")
    return value


@dataclass(frozen=True)
class ValidatedRelease:
    model: str
    task: str
    seed: int
    checkpoint_sha256: str
    evaluation_sha256: str
    dataset_manifest_sha256: str
    validation_mean_relative_l2: float


def validate_release(manifest_path: Path, checkpoint_path: Path) -> ValidatedRelease:
    """Verify an explicit approval and its checkpoint, evaluation, and data lineage."""
    manifest_path = Path(manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(f"cannot read release manifest: {exc}") from exc

    if manifest.get("schema_version") != "1.0":
        raise ReleaseValidationError("unsupported release manifest schema")
    if manifest.get("decision") != "approved":
        raise ReleaseValidationError("checkpoint has not been approved")
    if manifest.get("intended_use") != "research_case_inference":
        raise ReleaseValidationError("release is not approved for research case inference")

    checkpoint_sha = _required(manifest, "checkpoint_sha256", str)
    if _sha256(checkpoint_path) != checkpoint_sha:
        raise ReleaseValidationError("checkpoint SHA-256 does not match release manifest")

    base = manifest_path.parent
    evaluation_path = (base / _required(manifest, "evaluation_artifact", str)).resolve()
    dataset_manifest_path = (base / _required(manifest, "dataset_manifest", str)).resolve()
    evaluation_sha = _required(manifest, "evaluation_sha256", str)
    dataset_sha = _required(manifest, "dataset_manifest_sha256", str)
    if _sha256(evaluation_path) != evaluation_sha:
        raise ReleaseValidationError("evaluation artifact SHA-256 does not match")
    if _sha256(dataset_manifest_path) != dataset_sha:
        raise ReleaseValidationError("dataset manifest SHA-256 does not match")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("evidence_label") != "airfrans_official_split_result":
        raise ReleaseValidationError("evaluation is not an official-split result")
    if evaluation.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ReleaseValidationError("evaluation was produced by a different checkpoint")
    config = _required(evaluation, "config", dict)
    for name in ("model", "task", "seed"):
        if config.get(name) != manifest.get(name):
            raise ReleaseValidationError(f"release {name} does not match evaluation")
    splits = _required(evaluation, "splits", dict)
    if not all(
        isinstance(splits.get(name), list) and splits[name]
        for name in ("train", "validation", "test")
    ):
        raise ReleaseValidationError(
            "evaluation must record non-empty train/validation/test splits"
        )
    if set(splits["train"]) & set(splits["validation"]):
        raise ReleaseValidationError("training and validation splits overlap")
    if (set(splits["train"]) | set(splits["validation"])) & set(splits["test"]):
        raise ReleaseValidationError("protected test cases overlap model-development splits")

    validation = float(evaluation["best_validation_mean_relative_l2"])
    maximum = float(_required(manifest, "maximum_validation_mean_relative_l2", (int, float)))
    if validation > maximum:
        raise ReleaseValidationError("checkpoint exceeds the approved validation-error ceiling")

    return ValidatedRelease(
        model=str(manifest["model"]),
        task=str(manifest["task"]),
        seed=int(manifest["seed"]),
        checkpoint_sha256=checkpoint_sha,
        evaluation_sha256=evaluation_sha,
        dataset_manifest_sha256=dataset_sha,
        validation_mean_relative_l2=validation,
    )
