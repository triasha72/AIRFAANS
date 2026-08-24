# Real AirfRANS runbook

Use this runbook only after the deterministic CI fixture passes. It turns the
implemented pipeline into measured evidence; until then, report tables remain
pending.

## 1. Hardware and environment

Use a Linux host with an NVIDIA GPU. Record:

```bash
git rev-parse HEAD
python --version
nvidia-smi
python -m pip freeze > artifacts/local/pip-freeze.txt
```

Install `.[ml,airfrans,tracking,api,dev]` plus the official `airfrans` package.
Keep the CUDA, PyTorch, PyG, PyVista, Dask, and MLflow versions with the run.

## 2. Freeze the data

Download using the official package. Build a file manifest containing relative
path, byte count, and SHA-256. Record the AirfRANS task lists and do not redefine
the protected test cases after looking at results.

Inspect at least three raw OpenFOAM/VTK cases manually. Confirm coordinate units,
velocity and pressure conventions, inward/outward normal direction, surface
membership, cell connectivity, and the turbulent-viscosity units. Update the
normalization adapter only if the source contract requires it, with a regression
test for each mapping.

## 3. Preprocess

Convert cases to compressed `FlowCase` files. Use mesh connectivity as the main
graph and k-NN as a controlled alternative. Validate every case and save:

- simulation ID and source hash;
- Reynolds number and angle of attack;
- node and directed-edge counts;
- feature/target statistics;
- surface-node count;
- preprocessing version and elapsed time.

Dask may parallelize cases, but a serial run over a small fixed sample must
produce byte-identical arrays.

## 4. Train the matched comparison

Use `configs/experiment_v0_1.yaml`. For each model and seed:

1. Fit normalization on training simulations only.
2. Run one tiny overfit test on a training case.
3. Train with the same optimizer, schedule, early-stopping rule, and budget.
4. Resume once from a saved checkpoint.
5. Retain configuration, losses, best checkpoint hash, peak memory, and timing.
6. Evaluate once on the protected task test set.

Do not tune against Reynolds/AoA OOD test performance.

## 5. Evaluate physics and uncertainty

Report per-field RMSE, MAE, relative L2, and spatial error maps. Add pressure
coefficient and integrated lift/drag errors using the verified AirfRANS surface
normal and pressure conventions. Add viscous contribution before calling drag a
complete aerodynamic coefficient.

Train three independently seeded members for uncertainty. Report calibration,
uncertainty-error correlation, and the mean OOD/ID uncertainty ratio. A useful
OOD detector should rise under extrapolation; report honestly if it does not.

## 6. Active-learning comparison

Freeze an initial 10% simulation set. At each acquisition round select the same
number of new CFD cases using either mean ensemble field uncertainty or seeded
random selection. Retrain under equal compute budgets and plot test error against
the number of CFD simulations consumed. Repeat across seeds.

## 7. Performance and interface

Benchmark preprocessing throughput, nodes/s, edges/s, GPU peak memory, training
step time, and inference latency by mesh size. Compare inference with recorded
OpenFOAM solver wall time only when hardware and solver settings are documented.

Wire only the best validated checkpoint to the FastAPI predictor. Test `/health`,
valid inference, invalid input, missing geometry, and concurrent requests. Build
the Docker image and repeat the same smoke test in the container.

## 8. Publication gate

- CI and Docker checks pass on the published commit.
- Every number in the README and portfolio links to a raw artifact and report.
- Contour plots share color ranges and show CFD, prediction, and absolute error.
- Three-seed mean and variation are reported.
- Negative and inconclusive results remain in the report.
- A second reviewer audits split identities and at least five derived-force cases.
- Dataset authors and licenses are credited.
