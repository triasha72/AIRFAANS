# AIRFAANS: the remaining research work

[Read the project overview and measured results](README.md)

The interpolation study is complete, including three architectures, three
seeds, all 200 official test meshes, and force-coefficient checks. It also made
one point clear: no architecture wins on every output. MeshGraphNet produced the
lowest mean error across all four fields and drag at this budget, while the
point neural operator had the lowest mean lift error.

## The question to answer next

Can uncertainty tell us when the surrogate is leaving familiar aerodynamic
conditions?

Checkpoint ensembles, per-case uncertainty, OOD comparison, and case-selection
primitives are implemented. Full-mesh ensemble measurements, calibration, and
the matched active-learning retraining loop remain open. Single-model Reynolds
results reported in the README are not ensemble-UQ results. See
[the evidence audit](docs/evidence-gaps.md) for recovery and closure criteria.

## Planned run order

1. Evaluate the same frozen ensemble on interpolation, Reynolds OOD, and angle-
   of-attack OOD cases.
2. Compare uncertainty with actual field and lift/drag error for every case.
3. Repeat simulation selection with uncertainty and random acquisition using
   matched budgets and seeds.
4. Profile accuracy, memory, and inference time at useful mesh sizes.
5. Test another geometry family before discussing broad generalization.

A credible next release needs field metrics, engineering force metrics, seed
variation, and a useful relationship between uncertainty and error. A better
interpolation average by itself will not close this gap.

The readiness gate now checks usefulness, not file presence. Both OOD tasks
must raise mean ensemble uncertainty by at least 10%, uncertainty/error
correlation must reach 0.30, and uncertainty-guided acquisition must beat the
matched random arm. Until the official compute run supplies those values, the
release remains rejected.

## Reproducible execution

The remaining matrix is now executable with one command per treatment and
resumes from existing checkpoints:

```bash
python scripts/run_evidence_matrix.py \
  --dataset-root /path/to/airfrans \
  --output-root artifacts/evidence_matrix
```

Use `--dry-run` to print the exact commands before submitting them to a GPU
host. The script intentionally stops when the dataset, manifest, or config is
missing and never creates placeholder evidence. The default selects MeshGraphNet
only: nine task/seed treatments. Repeat with `--model pointwise_mlp` and
`--model point_neural_operator` for the full 27-treatment matrix, reusing only
compatible verified checkpoints. Then evaluate a fixed checkpoint ensemble on
interpolation, Reynolds-OOD, and AoA-OOD. Implement and run the matched
active-learning retraining comparison before invoking
`scripts/assess_operational_readiness.py`.
