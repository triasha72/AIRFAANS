# AirfRANS field-surrogate comparison v0.1

Status: first official full-mesh GPU treatment complete; matched benchmark in progress.

The first AirfRANS model-performance result is now recorded below. The official
1,000-case task manifest has been frozen, and one real simulation has traversed
VTK loading, field validation, surface-normal alignment, and mesh-graph
construction. The resulting 181,794-node, 724,640-edge artifact is tied to the
three source-file hashes in `artifacts/evaluation/airfrans_ingestion_v0_1.json`.

All three architectures also passed a bounded real-data optimization check on
the same simulation. The 512-node, 100-step CPU run reduced normalized training
MSE by 64.4% (MLP), 87.7% (MeshGraphNet), and 67.5% (point operator). These are
training-loss sanity checks, not held-out metrics or evidence that one model is
better than another.

## Official MeshGraphNet interpolation treatment

The seed-17 MeshGraphNet-style treatment requested 50 epochs on two Kaggle T4
GPUs for training. Its selected checkpoint reached validation mean relative L2
`0.6057466310` (best epoch 43). Evaluation was then run rank-zero on every node
of all 200 official interpolation test cases. Because full-mesh force recovery
can outlive an interactive session, evaluation was split into independently
persisted case records. Aggregation required all 200 official indices and one
common checkpoint hash.

| Metric | velocity-x | velocity-y | pressure | turbulent viscosity |
|---|---:|---:|---:|---:|
| Mean relative L2 | 0.401253 | 0.748747 | 0.826120 | 0.788552 |
| Mean RMSE | 19.820120 | 17.614057 | 1442.561367 | 0.001947 |
| Mean MAE | 14.899915 | 10.969271 | 828.576706 | 0.000947 |

Mean absolute force-coefficient error was `0.301217` for drag and `0.526824`
for lift. Full-test evaluation took `1,121.876 s`. The checkpoint SHA-256 is
`66e7b3bc19dc2a6582c80ddaab5a561d92029289d4d726fef6e24c01df140295`; the
aggregate `result.json` SHA-256 is
`4e674c7a32ea067526f3b170cb0416ddd74401a61a92284d4566a2c9f738b51a`.

These numbers show a functioning geometry-aware surrogate and expose where the
model is still weak: velocity-x is materially easier than cross-flow velocity,
pressure, and turbulent viscosity, and force errors are not yet small enough
for design use. The matched 50-epoch seed-17 pass is now complete for all three
architectures. MeshGraphNet leads pressure and turbulent-viscosity relative L2
and drag MAE; the Pointwise MLP leads both velocity components and lift MAE.
The compact point operator does not win a reported metric. Seeds 29 and 41 are
still required before treating this single-seed ordering as stable.

## Complete-pipeline coursework check

The complete experimental workflow has now been checked on real, disjoint
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

For the coursework study, this closes the workflow-integration gap: VTK
loading, disjoint splits, normalization, optimization, validation,
checkpointing, checkpoint reload, inference, and held-out metrics now execute
as one command. It does not close the full GPU benchmark gap, and it is not
evidence of production deployment.

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
| Full/interpolation | Pointwise MLP | 0.9355 | ux 0.3904; uy 0.7091 | 0.7888 | 0.2932 | 0.3628 | seed 17 only |
| Full/interpolation | MeshGraphNet | 0.8261 | ux 0.4013; uy 0.7487 | 0.7886 | 0.5268 | 0.3012 | seed 17 only |
| Full/interpolation | Point operator | 1.2434 | ux 0.4283; uy 1.0443 | 0.8018 | 0.2959 | 0.3695 | seed 17 only |
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
