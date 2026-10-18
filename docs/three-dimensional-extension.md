# 3-D extension: current boundary and required work

The published implementation is 2-D. `FlowCase.validate` requires `[nodes, 2]`
positions and normals, `GraphCase` expects `(dx, dy, distance)` edge attributes,
and the field contract is `(u_x, u_y, pressure, turbulent_viscosity)`. The
AirfRANS ingestion and force calculation also use the 2-D benchmark convention.
Generic relative-position code does not establish a working 3-D CFD pipeline.

An older local checkout contains unpublished interface changes. They are not
part of the published baseline and require separate review and verification.

Before a 3-D implementation can be claimed:

1. Declare a 3-D dataset, units, mesh-cell types, boundary conditions, and split
   policy, including held-out geometries.
2. Extend coordinates, normals, velocity targets, graph edge widths, model input
   and output widths, normalization, saved-case formats, and API validation
   together. Preserve compatibility explicitly or version the serialized format.
3. Implement a verified 3-D pressure and viscous surface-force integral, including
   oriented face areas and a declared lift/drag reference frame.
4. Test actual 3-D ingestion and graph connectivity, then overfit a small real
   case before running a matched held-out benchmark.
5. Publish per-case field/force errors, independent seeds, checkpoint and data
   hashes, hardware, memory, and inference timing.

A synthetic 3-D graph test would validate interfaces only. It would not provide
3-D flow accuracy, rotational equivariance, or Galilean invariance evidence.

Frame consistency is a separate research task. A useful first experiment would
declare the freestream/body reference frame, transform velocity features,
targets and boundary conditions consistently, and compare predictions after
transforming them back. A uniform velocity offset at fixed coordinates is not
by itself a complete Galilean transformation of a steady boundary-value problem.
Keep any such result labeled as a frame-consistency study with explicit limits.
