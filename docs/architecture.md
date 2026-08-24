# Architecture decisions

## Goal

Predict complete aerodynamic fields on irregular meshes while preserving the
engineering quantities derived from them. The system must also expose where it
fails under changes in geometry and operating condition.

## Representation

One `FlowCase` is one simulation. Nodes contain coordinates, freestream
velocity, distance to the surface, surface normals, log Reynolds number, and
angle of attack. Targets are velocity-x, velocity-y, pressure, and turbulent
viscosity. Simulation identity is retained through preprocessing and splitting.

The primary graph uses solver mesh cells. A k-NN graph is retained as a
controlled point-cloud treatment for processed datasets without connectivity.
Edges carry relative x/y displacement and Euclidean distance. This makes the
geometric assumption visible and testable.

## Model comparison

The pointwise MLP asks how far local features alone can go. The MeshGraphNet
adds local information exchange along edges. The point operator adds global
interaction through learned modes. This is a three-level comparison of local,
local-plus-neighborhood, and global field mappings; it is not an architecture
search.

## Data leakage boundary

Nodes from the same CFD simulation are correlated. AIRFAANS therefore splits by
simulation case. Normalization statistics are fit on training simulations only.
Protected test cases cannot be used to choose graph construction, features,
model depth, early stopping, or uncertainty calibration.

## Experiment execution

The YAML-driven runner resolves one model, task, and seed at a time. It fits
streaming feature and target moments from training cases, samples boundary and
volume nodes deterministically, evaluates a fixed validation partition, applies
learning-rate reduction and early stopping, and reloads the best checkpoint
before testing. Bounded case-count overrides receive a different evidence label
from official-split runs. Official runs evaluate every test-mesh node; sampled
metrics cannot silently enter the benchmark table.

Complete-mesh evaluation uses the solver connectivity for MeshGraphNet and
reproduces the official AirfRANS pressure and molecular-viscous force
integration. This requires full-node predictions because wall shear is derived
from the VTK velocity gradient. Consequently, the API and bounded sampled runs
return no CL/CD instead of estimating it from an incomplete surface.

## Failure behavior

Missing VTK arrays, invalid graphs, split overlap, absent checkpoints, and
non-finite fields are errors. The API is unavailable until a validated model is
configured. Unsupported framework or hardware combinations are reported as
unsupported, not as zero-valued measurements.
