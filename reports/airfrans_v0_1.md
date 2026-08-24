# AirfRANS field-surrogate comparison v0.1

Status: real-data ingestion verified; GPU model training and evaluation pending.

No AirfRANS model-performance result is claimed in this report yet. The official
1,000-case task manifest has been frozen, and one real simulation has traversed
VTK loading, field validation, surface-normal alignment, and mesh-graph
construction. The resulting 181,794-node, 724,640-edge artifact is tied to the
three source-file hashes in `artifacts/evaluation/airfrans_ingestion_v0_1.json`.

All three architectures also passed a bounded real-data optimization check on
the same simulation. The 512-node, 100-step CPU run reduced normalized training
MSE by 64.4% (MLP), 87.7% (MeshGraphNet), and 67.5% (point operator). These are
training-loss sanity checks, not held-out metrics or evidence that one model is
better than another.

## End-to-end bounded run

The production experiment path has now been executed on real, disjoint
simulations: pointwise MLP, seed 17, CPU, two training cases, one validation
case, one official full-test case, 256 stratified nodes per case, and three
epochs. It used train-only streaming normalization, validation-based checkpoint
selection, and checkpoint reload. The selected checkpoint SHA-256 is
`3896c6d86e28670bab62d36b24d23ccff71163480b8b530ea06a4303137449de`.

The best validation mean relative L2 was 0.8484. On the single sampled test
case, relative L2 was 0.6413 for velocity-x, 1.3318 for velocity-y, 1.5476 for
pressure, and 1.0221 for turbulent viscosity. These poor values are expected
from the intentionally tiny budget. The artifact label is
`airfrans_bounded_run`; these numbers must not be copied into the benchmark
table or presented as successful surrogate performance.

This closes the software-integration gap: VTK loading, disjoint splits,
normalization, optimization, validation, checkpointing, checkpoint reload,
inference, and held-out metrics now execute as one command. It does not close
the full GPU benchmark gap.

## Force-convention verification

The complete-mesh evaluator now mirrors the official AirfRANS implementation:
it computes the VTK velocity gradient, deviatoric strain, molecular-viscous wall
stress, surface pressure integration, freestream-axis rotation, and dynamic
pressure normalization. All six components for a representative case matched
the official library to floating-point precision. Five reference cases are
frozen in `artifacts/evaluation/airfrans_force_verification_v0_1.json`.

Force coefficients remain absent from bounded sampled-node runs because those
samples cannot recover a trustworthy wall gradient. Full official runs evaluate
every mesh node and report total, pressure, and viscous CL/CD contributions.

## Questions

1. Does mesh message passing improve field prediction over a pointwise model?
2. Does the point operator handle long-range field dependence differently?
3. How much does each model degrade under Reynolds and angle-of-attack shifts?
4. Does ensemble uncertainty increase where field error increases and under OOD?
5. Do lower field errors translate into lower CL/CD errors?
6. Can uncertainty-guided case selection reduce the CFD cases required?

## Pending results

| Task | Model | Pressure rel. L2 | Velocity rel. L2 | nu_t rel. L2 | CL MAE | CD MAE | Mean ± variation |
|---|---|---:|---:|---:|---:|---:|---|
| Full/interpolation | Pointwise MLP | Pending | Pending | Pending | Pending | Pending | Pending |
| Full/interpolation | MeshGraphNet | Pending | Pending | Pending | Pending | Pending | Pending |
| Full/interpolation | Point operator | Pending | Pending | Pending | Pending | Pending | Pending |
| Reynolds OOD | Pointwise MLP | Pending | Pending | Pending | Pending | Pending | Pending |
| Reynolds OOD | MeshGraphNet | Pending | Pending | Pending | Pending | Pending | Pending |
| Reynolds OOD | Point operator | Pending | Pending | Pending | Pending | Pending | Pending |
| AoA OOD | Pointwise MLP | Pending | Pending | Pending | Pending | Pending | Pending |
| AoA OOD | MeshGraphNet | Pending | Pending | Pending | Pending | Pending | Pending |
| AoA OOD | Point operator | Pending | Pending | Pending | Pending | Pending | Pending |

## Evidence required before completion

- AirfRANS data manifest and exact official task splits.
- Training configurations, logs, three seeds, and checkpoint hashes.
- Raw per-case predictions and metrics.
- Predicted full-mesh CL/CD errors using the verified force convention.
- OOD uncertainty and active-learning curves.
- Named hardware, package versions, memory, throughput, and latency.
- Reproducible plots tied to the raw artifact hashes.
