# AirfRANS field-surrogate comparison v0.1

Status: matched three-seed interpolation benchmark complete; OOD studies pending.

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
The compact point operator does not win a reported metric. This was the
seed-17 snapshot; the completed three-seed result below supersedes it for the
current architecture comparison.

The seed-29 MeshGraphNet repeat is also complete. It produced full-mesh mean
relative L2 values of `0.36349` (velocity-x), `0.65474` (velocity-y), `0.71475`
(pressure), and `0.64414` (turbulent viscosity), with drag MAE `0.21254` and
lift MAE `0.19944`. All 200 official indices were aggregated from case records
tied to checkpoint SHA-256
`eba0c367474a784e1bb79731140788e90785f573f6d70974e8d81130ec714a25`.
Seed 41 and the remaining seed-29 architecture treatment are still required;
therefore no three-seed mean or variance is reported yet.

The seed-29 point neural operator subsequently completed all 200 official test
meshes. Mean relative L2 was `0.41084` (velocity-x), `1.03226` (velocity-y),
`1.19613` (pressure), and `0.80435` (turbulent viscosity); drag and lift MAE
were `0.32790` and `0.29857`. The result is tied to checkpoint SHA-256
`c9f0a44814804f767ff96aade6f924bdba119035efe38ab120d139c5c153cc05`.
This completes the matched seed-29 architecture pass, but seed 41 remains
pending for all three models.

## Three-seed matched interpolation result

The 50-epoch comparison is complete for seeds 17, 29, and 41. Each treatment
was evaluated on every node of all 200 official interpolation test cases.
Values are mean ± sample standard deviation across seeds.

| Model | ux rel. L2 | uy rel. L2 | pressure rel. L2 | nu_t rel. L2 | CD MAE | CL MAE |
|---|---:|---:|---:|---:|---:|---:|
| Pointwise MLP | 0.3993 ± 0.0141 | 0.7242 ± 0.0230 | 0.9550 ± 0.0314 | 0.7979 ± 0.0219 | 0.3663 ± 0.0221 | 0.3546 ± 0.1460 |
| MeshGraphNet | **0.3788 ± 0.0199** | **0.6703 ± 0.0720** | **0.7693 ± 0.0557** | **0.6895 ± 0.0859** | **0.2319 ± 0.0619** | 0.3907 ± 0.1705 |
| Point neural operator | 0.4183 ± 0.0090 | 1.0314 ± 0.0133 | 1.2125 ± 0.0268 | 0.8127 ± 0.0167 | 0.3544 ± 0.0230 | **0.2843 ± 0.0225** |

The geometry-aware MeshGraphNet is the strongest field surrogate under this
matched coursework contract and also has the lowest drag error. The compact
point operator produces the lowest lift MAE but does not match MeshGraphNet on
the field quantities. Lift error is notably seed-sensitive for both the MLP and
MeshGraphNet. This supports a bounded conclusion about the tested interpolation
setting, not a universal model ranking.

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
