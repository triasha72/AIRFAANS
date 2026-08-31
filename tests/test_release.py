import hashlib
import json
from pathlib import Path

import pytest

from airfaans.release import ReleaseValidationError, validate_release


def _write_release(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"real checkpoint bytes")
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text('{"dataset":"AirfRANS"}\n')
    evaluation = tmp_path / "result.json"
    evaluation_payload = {
        "evidence_label": "airfrans_official_split_result",
        "config": {"model": "mesh_graph_net", "task": "interpolation", "seed": 17},
        "splits": {"train": ["train-a"], "validation": ["val-a"], "test": ["test-a"]},
        "best_validation_mean_relative_l2": 0.6,
        "checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
    }
    evaluation.write_text(json.dumps(evaluation_payload) + "\n")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "decision": "approved",
                "intended_use": "research_case_inference",
                "model": "mesh_graph_net",
                "task": "interpolation",
                "seed": 17,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "evaluation_artifact": "result.json",
                "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
                "dataset_manifest": "dataset.json",
                "dataset_manifest_sha256": hashlib.sha256(
                    dataset_manifest.read_bytes()
                ).hexdigest(),
                "maximum_validation_mean_relative_l2": 0.7,
            }
        )
        + "\n"
    )
    return release, checkpoint, evaluation


def test_release_validates_checkpoint_evaluation_and_dataset_lineage(tmp_path: Path):
    release, checkpoint, _ = _write_release(tmp_path)
    validated = validate_release(release, checkpoint)
    assert validated.model == "mesh_graph_net"
    assert validated.validation_mean_relative_l2 == 0.6


def test_release_rejects_checkpoint_tampering(tmp_path: Path):
    release, checkpoint, _ = _write_release(tmp_path)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ReleaseValidationError, match="checkpoint SHA-256"):
        validate_release(release, checkpoint)


def test_release_rejects_bounded_evaluation(tmp_path: Path):
    release, checkpoint, evaluation = _write_release(tmp_path)
    payload = json.loads(evaluation.read_text())
    payload["evidence_label"] = "airfrans_bounded_run"
    evaluation.write_text(json.dumps(payload) + "\n")
    manifest = json.loads(release.read_text())
    manifest["evaluation_sha256"] = hashlib.sha256(evaluation.read_bytes()).hexdigest()
    release.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ReleaseValidationError, match="official-split"):
        validate_release(release, checkpoint)
