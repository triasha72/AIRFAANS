# AIRFAANS project overview

## The problem

CFD is expensive enough that an engineer may want a learned surrogate, but a
low average field error is not enough. The surrogate also has to be judged on
new simulations and on lift and drag, not only on per-node values.

## What I built

I put a pointwise MLP, a MeshGraphNet-style GNN, and a point neural operator
behind one AirfRANS data and evaluation contract. The comparison uses
simulation-level splits, train-only normalization, three matched seeds, and all
200 official interpolation meshes per treatment. I also checked the pressure
and viscous-force convention against the AirfRANS implementation.

## What the evidence says

Across the completed three-seed interpolation study, MeshGraphNet had the
lowest mean error for the four reported fields and drag. The point neural
operator had the lowest mean lift error. This is a result for one split and
training budget, not a claim that one architecture wins every engineering task.

## Reproduce the software check

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
airfaans demo
pytest -q
```

The demo is an analytic fixture for exercising the software path. It is not an
AirfRANS or aerodynamic-performance result. The real-data setup and frozen
artifacts are documented in the [README](../README.md).

## Next validation

The important unfinished work is Reynolds-number and angle-of-attack OOD
evaluation, uncertainty calibration, and uncertainty-guided simulation
selection. Those experiments are deliberately kept outside the current result
claims.
