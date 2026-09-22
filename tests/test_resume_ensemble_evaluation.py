"""Tests for the restart-safe ensemble-evaluation wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from airfaans.uq_experiment import EnsembleManifest


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "resume_ensemble_evaluation.py"
    spec = importlib.util.spec_from_file_location("resume_ensemble_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plan_next_shard_starts_at_zero(tmp_path: Path):
    runner = _runner_module()
    manifest = EnsembleManifest(
        model="pointwise_mlp",
        training_task="reynolds_ood",
        seeds=(29, 41),
        checkpoint_paths=("a.pt", "b.pt"),
        checkpoint_sha256=("a", "b"),
    )
    state = runner.plan_next_shard(tmp_path / "ensemble_cases", manifest, "reynolds_ood", 4)

    assert state["saved_cases"] == 0
    assert state["next_missing_index"] == 0
    assert (tmp_path / "resume_state.json").exists()
