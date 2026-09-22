import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from airfaans.uq_experiment import (
    EnsembleManifest,
    audit_ensemble_records,
    compare_ood_uncertainty,
    load_ensemble_manifest,
    summarize_uq_cases,
)


def _manifest() -> EnsembleManifest:
    return EnsembleManifest(
        model="pointwise_mlp",
        training_task="reynolds_ood",
        seeds=(29, 41),
        checkpoint_paths=("first.pt", "second.pt"),
        checkpoint_sha256=("first-hash", "second-hash"),
    )


def _write_record(path: Path, manifest: EnsembleManifest, index: int) -> None:
    path.write_text(
        __import__("json").dumps(
            {
                "official_test_index": index,
                "evaluation_task": "reynolds_ood",
                "model": manifest.model,
                "training_task": manifest.training_task,
                "seeds": list(manifest.seeds),
                "checkpoint_sha256": list(manifest.checkpoint_sha256),
            }
        )
    )


def test_ensemble_audit_finds_first_missing_index(tmp_path: Path):
    records = tmp_path / "ensemble_cases"
    records.mkdir()
    manifest = _manifest()
    _write_record(records / "0.json", manifest, 0)
    _write_record(records / "2.json", manifest, 2)

    assert audit_ensemble_records(records, manifest, "reynolds_ood", 4).next_missing_index == 1


def test_ensemble_audit_rejects_changed_checkpoint_hash(tmp_path: Path):
    records = tmp_path / "ensemble_cases"
    records.mkdir()
    manifest = _manifest()
    _write_record(records / "0.json", manifest, 0)
    changed = replace(manifest, checkpoint_sha256=("other", "second-hash"))

    with pytest.raises(ValueError, match="mixed-provenance"):
        audit_ensemble_records(records, changed, "reynolds_ood", 4)


def _checkpoint(path: Path, *, seed: int, model: str = "pointwise_mlp") -> dict[str, object]:
    path.write_bytes(f"{seed}-{model}".encode())
    return {"config": {"seed": seed, "model": model, "task": "reynolds_ood"}}


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, payloads: dict[Path, dict[str, object]]) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(load=lambda path, **_: payloads[Path(path)]),
    )


def test_manifest_rejects_duplicate_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = tmp_path / "seed29.pt"
    second = tmp_path / "seed29-copy.pt"
    payloads = {first: _checkpoint(first, seed=29), second: _checkpoint(second, seed=29)}
    _install_fake_torch(monkeypatch, payloads)

    with pytest.raises(ValueError, match="distinct seeds"):
        load_ensemble_manifest([first, second])


def test_manifest_rejects_mixed_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = tmp_path / "mlp.pt"
    second = tmp_path / "gnn.pt"
    payloads = {
        first: _checkpoint(first, seed=29, model="pointwise_mlp"),
        second: _checkpoint(second, seed=41, model="mesh_graph_net"),
    }
    _install_fake_torch(monkeypatch, payloads)

    with pytest.raises(ValueError, match="share model and training task"):
        load_ensemble_manifest([first, second])


def test_uq_case_summary_and_ood_comparison():
    cases = [
        {"mean_uncertainty": 1.0, "uncertainty_error_correlation": 0.2},
        {"mean_uncertainty": 2.0, "uncertainty_error_correlation": 0.4},
    ]
    assert summarize_uq_cases(cases) == {
        "mean_uncertainty": 1.5,
        "mean_uncertainty_error_correlation": pytest.approx(0.3),
    }
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b", "c"],
        "summary": {"mean_uncertainty": 1.5},
    }
    shifted = {
        "evaluation_task": "reynolds_ood",
        "checkpoint_sha256": ["a", "b", "c"],
        "summary": {"mean_uncertainty": 2.25},
    }
    comparison = compare_ood_uncertainty(base, shifted)
    assert comparison["ood_to_id_uncertainty_ratio"] == 1.5
    assert comparison["passed_predeclared_ratio"]
