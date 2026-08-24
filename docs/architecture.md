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

## Failure behavior

Missing VTK arrays, invalid graphs, split overlap, absent checkpoints, and
non-finite fields are errors. The API is unavailable until a validated model is
configured. Unsupported framework or hardware combinations are reported as
unsupported, not as zero-valued measurements.
