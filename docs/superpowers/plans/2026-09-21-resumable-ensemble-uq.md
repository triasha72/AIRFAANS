# Resumable ensemble UQ implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add restart-safe full-mesh ensemble evaluation that produces provenance-checked ID and OOD uncertainty evidence.

**Architecture:** A checkpoint manifest freezes ensemble identity before inference. The evaluator writes one atomic record per official case and validates records on restart. Aggregation only writes a report after complete coverage.

**Tech Stack:** Python, PyTorch, NumPy, JSON, pytest, AirfRANS official task manifests.

**Spec:** `docs/superpowers/specs/2026-09-21-airfaans-uq-active-learning-design.md`

## Global Constraints

- Require at least two checkpoints with distinct seeds and one shared model and training task.
- Save SHA-256 values for every checkpoint and reject identity mismatch before inference.
- Write per-case JSON atomically; only reuse a record with matching complete provenance.
- Aggregate only when every expected official index appears exactly once.
- Label a two-member ensemble by its real size; do not call it a broad calibration study.

## Review Focus

- A checkpoint can change after a partial run. Task 1 rejects a changed SHA-256.
- A valid JSON record can belong to another task. Task 2 rejects mixed provenance.
- A runtime can stop during a write. Task 2 ignores incomplete temporary files.
- Saved indexes can be non-contiguous. Task 2 finds the first missing index.
- ID and OOD reports can reorder hashes. Task 3 rejects that comparison.

---

## File structure

- Modify `src/airfaans/uq_experiment.py`: manifest, audit, shard evaluation, aggregation.
- Modify `src/airfaans/cli.py`: shard and aggregate commands.
- Modify `tests/test_uq_experiment.py`: contracts for all UQ components.
- Create `scripts/resume_ensemble_evaluation.py`: restart wrapper.
- Create `tests/test_resume_ensemble_evaluation.py`: wrapper preflight tests.
- Modify `docs/real-data-runbook.md`: persistent-storage commands.

### Task 1: Freeze ensemble identity

**Files:** Modify `src/airfaans/uq_experiment.py` and `tests/test_uq_experiment.py`.

**Interfaces:** Produce `EnsembleManifest(model: str, training_task: str, seeds: tuple[int, ...], checkpoint_paths: tuple[str, ...], checkpoint_sha256: tuple[str, ...])` and `load_ensemble_manifest(checkpoint_paths: list[Path]) -> EnsembleManifest`.

- [ ] **Step 1: Write failing identity tests**

```python
def test_manifest_rejects_duplicate_seed(tmp_path: Path):
    paths = [write_checkpoint(tmp_path / "a.pt", seed=29), write_checkpoint(tmp_path / "b.pt", seed=29)]
    with pytest.raises(ValueError, match="distinct seeds"):
        load_ensemble_manifest(paths)

def test_manifest_rejects_mixed_model(tmp_path: Path):
    paths = [write_checkpoint(tmp_path / "a.pt", seed=29, model="pointwise_mlp"), write_checkpoint(tmp_path / "b.pt", seed=41, model="mesh_graph_net")]
    with pytest.raises(ValueError, match="share model and training task"):
        load_ensemble_manifest(paths)
```

- [ ] **Step 2: Run `pytest tests/test_uq_experiment.py -k manifest -v`.** Expected: FAIL because `load_ensemble_manifest` is undefined.

- [ ] **Step 3: Implement the manifest boundary.** Load every serialized checkpoint config, calculate its SHA-256, reject fewer than two paths, repeated seeds, and a non-singleton `(config.model, config.task)` set before returning the immutable dataclass.

- [ ] **Step 4: Run `pytest tests/test_uq_experiment.py -k manifest -v`.** Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add src/airfaans/uq_experiment.py tests/test_uq_experiment.py && git commit -m "feat: freeze ensemble checkpoint identity"`.

### Task 2: Persist and resume per-case ensemble evidence

**Files:** Modify `src/airfaans/uq_experiment.py` and `tests/test_uq_experiment.py`.

**Interfaces:** Produce `audit_ensemble_records(records_dir: Path, manifest: EnsembleManifest, evaluation_task: str, expected_cases: int) -> EnsembleAudit` and `evaluate_ensemble_shard(..., start: int, count: int | None, resume: bool) -> dict[str, object]`. Records live in `output_dir / "ensemble_cases" / "{official_test_index}.json"`.

- [ ] **Step 1: Write failing audit tests**

```python
def test_audit_finds_first_missing_index(tmp_path: Path, manifest: EnsembleManifest):
    records = tmp_path / "ensemble_cases"; records.mkdir()
    write_record(records / "0.json", manifest, task="reynolds_ood", index=0)
    write_record(records / "2.json", manifest, task="reynolds_ood", index=2)
    assert audit_ensemble_records(records, manifest, "reynolds_ood", 4).next_missing_index == 1

def test_audit_rejects_changed_hash(tmp_path: Path, manifest: EnsembleManifest):
    records = tmp_path / "ensemble_cases"; records.mkdir()
    write_record(records / "0.json", manifest, task="reynolds_ood", index=0)
    changed = replace(manifest, checkpoint_sha256=("other", *manifest.checkpoint_sha256[1:]))
    with pytest.raises(ValueError, match="mixed-provenance"):
        audit_ensemble_records(records, changed, "reynolds_ood", 4)
```

- [ ] **Step 2: Run `pytest tests/test_uq_experiment.py -k audit -v`.** Expected: FAIL because `audit_ensemble_records` is undefined.

- [ ] **Step 3: Implement atomic records and shard evaluation.** Write to `path.with_suffix(".json.tmp")`, then replace the final path. Validate task, model, training task, seed list, hash list, bounds, and duplicate indexes before inferring only requested official indexes.

- [ ] **Step 4: Run `pytest tests/test_uq_experiment.py -k audit -v`.** Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add src/airfaans/uq_experiment.py tests/test_uq_experiment.py && git commit -m "feat: add resumable ensemble case records"`.

### Task 3: Aggregate complete reports and enforce ID/OOD identity

**Files:** Modify `src/airfaans/uq_experiment.py` and `tests/test_uq_experiment.py`.

**Interfaces:** Produce `aggregate_ensemble_records(output_dir: Path, manifest: EnsembleManifest, evaluation_task: str, expected_cases: int) -> dict[str, object]`; write `output_dir / "ensemble_result.json"` only after complete coverage.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_aggregate_requires_complete_coverage(tmp_path: Path, manifest: EnsembleManifest):
    with pytest.raises(ValueError, match="incomplete coverage"):
        aggregate_ensemble_records(tmp_path, manifest, "interpolation", expected_cases=2)

def test_compare_rejects_reordered_hashes():
    base = {"evaluation_task": "interpolation", "checkpoint_sha256": ["a", "b"], "summary": {"mean_uncertainty": 1.0}}
    shifted = {"evaluation_task": "reynolds_ood", "checkpoint_sha256": ["b", "a"], "summary": {"mean_uncertainty": 1.2}}
    with pytest.raises(ValueError, match="same ensemble checkpoints"):
        compare_ood_uncertainty(base, shifted)
```

- [ ] **Step 2: Run `pytest tests/test_uq_experiment.py -k "aggregate or reordered" -v`.** Expected: FAIL because aggregation is undefined and comparison allows reordered hashes.

- [ ] **Step 3: Implement aggregation.** Re-run record audit; raise `ValueError("incomplete coverage; aggregate report not written")` when an index is missing. Read records in official-index order, summarize them, and atomically write `ensemble_result.json` with `ensemble_size` and `bounded=False`. Require ordered checkpoint-hash equality in `compare_ood_uncertainty`.

- [ ] **Step 4: Run `pytest tests/test_uq_experiment.py -k "aggregate or reordered" -v`.** Expected: PASS.

- [ ] **Step 5: Commit.** Run `git add src/airfaans/uq_experiment.py tests/test_uq_experiment.py && git commit -m "feat: aggregate verified ensemble uncertainty evidence"`.

### Task 4: Expose the restart-safe CLI and runbook

**Files:** Create `scripts/resume_ensemble_evaluation.py` and `tests/test_resume_ensemble_evaluation.py`; modify `src/airfaans/cli.py` and `docs/real-data-runbook.md`.

**Interfaces:** Produce `plan_next_shard(...) -> dict[str, object]` with saved count, expected count, missing index, and ordered hashes. Add `evaluate-ensemble` options `--output-dir`, `--start`, `--count`, `--no-resume`, plus `aggregate-ensemble`.

- [ ] **Step 1: Write the failing wrapper test**

```python
def test_plan_next_shard_starts_at_zero(tmp_path: Path):
    state = runner.plan_next_shard(tmp_path / "ensemble_cases", manifest_fixture(), "reynolds_ood", 4)
    assert state["saved_cases"] == 0
    assert state["next_missing_index"] == 0
```

- [ ] **Step 2: Run `pytest tests/test_resume_ensemble_evaluation.py -v`.** Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement the wrapper, CLI, and runbook.** The wrapper must write `resume_state.json`. The runbook must use mounted Drive or similar persistent storage, separate output directories per task, and a bounded smoke command before each full task command.

- [ ] **Step 4: Run `pytest tests/test_uq_experiment.py tests/test_resume_ensemble_evaluation.py tests/test_active_learning.py -v`.** Expected: PASS with no test failure.

- [ ] **Step 5: Commit.** Run `git add src/airfaans/uq_experiment.py src/airfaans/cli.py scripts/resume_ensemble_evaluation.py tests/test_uq_experiment.py tests/test_resume_ensemble_evaluation.py docs/real-data-runbook.md && git commit -m "feat: expose resumable ensemble evaluation commands"`.

## Follow-on plan

After a verified UQ report exists, write a separate active-learning plan that adds protected evaluation manifests, per-round retraining receipts, and matched random-arm reporting on top of the immutable UQ report.
