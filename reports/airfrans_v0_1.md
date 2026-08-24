# AirfRANS field-surrogate comparison v0.1

Status: preregistered; real-data and GPU execution pending.

No AirfRANS performance result is claimed in this report yet. The analytic CI
fixture validates interfaces and deterministic calculations only.

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
- Verified pressure/normal/viscous-force conventions.
- OOD uncertainty and active-learning curves.
- Named hardware, package versions, memory, throughput, and latency.
- Reproducible plots tied to the raw artifact hashes.
