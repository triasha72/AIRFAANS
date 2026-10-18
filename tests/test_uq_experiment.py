import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from airfaans.uq_experiment import (
    EnsembleManifest,
    aggregate_ensemble_records,
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


def test_aggregate_ensemble_requires_complete_coverage(tmp_path: Path):
    with pytest.raises(ValueError, match="incomplete coverage"):
        aggregate_ensemble_records(tmp_path, _manifest(), "interpolation", expected_cases=2)


def test_aggregate_reads_records_by_official_index_not_filename(tmp_path: Path):
    manifest = _manifest()
    records = tmp_path / "ensemble_cases"
    records.mkdir()
    for index, filename in ((0, "first.json"), (1, "second.json")):
        _write_record(records / filename, manifest, index)
        payload = __import__("json").loads((records / filename).read_text())
        payload.update({"mean_uncertainty": float(index + 1), "uncertainty_error_correlation": 0.5})
        (records / filename).write_text(__import__("json").dumps(payload))

    result = aggregate_ensemble_records(tmp_path, manifest, "reynolds_ood", expected_cases=2)

    assert result["case_count"] == 2
    assert result["summary"]["mean_uncertainty"] == 1.5


def test_compare_ood_rejects_reordered_hashes():
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b"],
        "summary": {"mean_uncertainty": 1.0},
    }
    shifted = {
        "evaluation_task": "reynolds_ood",
        "checkpoint_sha256": ["b", "a"],
        "summary": {"mean_uncertainty": 1.2},
    }
    with pytest.raises(ValueError, match="same ensemble checkpoints"):
        compare_ood_uncertainty(base, shifted)


def _checkpoint(path: Path, *, seed: int, model: str = "pointwise_mlp") -> dict[str, object]:
    path.write_bytes(f"{seed}-{model}".encode())
    return {"config": {"seed": seed, "model": model, "task": "reynolds_ood"}}


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch, payloads: dict[Path, dict[str, object]]
) -> None:
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


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_uq_summary_rejects_invalid_uncertainty(invalid):
    with pytest.raises(ValueError, match="uncertainty"):
        summarize_uq_cases([{"mean_uncertainty": invalid, "uncertainty_error_correlation": 0.5}])


def test_ood_comparison_uses_documented_ten_percent_threshold():
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b"],
        "summary": {"mean_uncertainty": 1.0},
    }
    shifted = {**base, "evaluation_task": "aoa_ood", "summary": {"mean_uncertainty": 1.05}}
    assert not compare_ood_uncertainty(base, shifted)["passed_predeclared_ratio"]


def test_ood_comparison_rejects_bounded_evidence():
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b"],
        "bounded": True,
        "summary": {"mean_uncertainty": 1.0},
    }
    shifted = {**base, "evaluation_task": "aoa_ood"}
    with pytest.raises(ValueError, match="bounded"):
        compare_ood_uncertainty(base, shifted)


def test_ood_comparison_rejects_an_id_report_used_as_ood():
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b"],
        "summary": {"mean_uncertainty": 1.0},
    }
    with pytest.raises(ValueError, match="task"):
        compare_ood_uncertainty(base, base)


@pytest.mark.parametrize("max_cases", [None, 1])
def test_legacy_ensemble_evaluates_full_meshes(tmp_path, monkeypatch, max_cases):
    import json
    from dataclasses import asdict

    import numpy as np
    import torch

    import airfaans.uq_experiment as uq
    from airfaans.experiment import ExperimentConfig, build_model
    from airfaans.normalization import Normalization
    from airfaans.synthetic import make_case

    case = make_case(radial_count=2, angular_count=8)
    normalization = Normalization(np.zeros(9), np.ones(9), np.zeros(4), np.ones(4))
    checkpoints = []
    for seed in (29, 41):
        config = ExperimentConfig(
            model="pointwise_mlp",
            seed=seed,
            hidden_dim=4,
            nodes_per_case=8,
            validation_cases=1,
        )
        path = tmp_path / f"seed-{seed}.pt"
        torch.save(
            {
                "config": asdict(config),
                "normalization": normalization.to_dict(),
                "model": build_model(config, 9).state_dict(),
            },
            path,
        )
        checkpoints.append(path)
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {"splits": {"full_train": ["train", "validation"], "full_test": ["case-a", "case-b"]}}
        )
    )
    # Replace only external VTK I/O; use actual checkpoints, inference and metrics.
    monkeypatch.setattr(uq, "load_case", lambda _: case)
    monkeypatch.setattr(uq, "case_directory", lambda root, case_id: root / case_id)
    result = uq.evaluate_ensemble(
        tmp_path,
        manifest,
        checkpoints,
        "interpolation",
        tmp_path / "result.json",
        max_cases=max_cases,
    )
    assert result["per_case"][0]["field_metrics"]["node_count"] == 16
    assert result["bounded"] is (max_cases is not None)
    assert result["case_count"] == (2 if max_cases is None else 1)


@pytest.mark.parametrize("ratio, passed", [(1.09, False), (1.1, True)])
def test_ood_comparison_threshold_boundary(ratio, passed):
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b"],
        "summary": {"mean_uncertainty": 1.0},
    }
    shifted = {**base, "evaluation_task": "aoa_ood", "summary": {"mean_uncertainty": ratio}}
    assert compare_ood_uncertainty(base, shifted)["passed_predeclared_ratio"] is passed


def test_ood_comparison_rejects_overflowing_ratio():
    base = {
        "evaluation_task": "interpolation",
        "checkpoint_sha256": ["a", "b"],
        "summary": {"mean_uncertainty": 1e-308},
    }
    shifted = {**base, "evaluation_task": "aoa_ood", "summary": {"mean_uncertainty": 1e308}}
    with pytest.raises(ValueError, match="finite"):
        compare_ood_uncertainty(base, shifted)
