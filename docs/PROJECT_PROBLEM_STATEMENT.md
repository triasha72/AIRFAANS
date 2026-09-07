# AIRFAANS problem statement

## Design need

Aerodynamic design studies often require many CFD simulations across airfoil
geometries and operating conditions. Full simulations provide the flow fields
and force coefficients engineers need, but their cost makes broad design-space
exploration slow.

A learned surrogate could shorten that loop only if it produces useful field
estimates on complete, irregular meshes and does not conceal poor lift, drag,
or out-of-distribution behaviour behind a single average error.

## Research question

For the official AirfRANS interpolation task, how do a pointwise MLP, a
MeshGraphNet-style GNN, and a point neural operator compare when they are
trained and evaluated under the same data, split, budget, and force-calculation
contract?

## Scope

AIRFAANS predicts velocity-x, velocity-y, pressure, and turbulent viscosity at
each node of an airfoil CFD mesh. It uses geometry and operating-condition
features, then evaluates both field error and integrated lift and drag error.

The project began as Georgia Tech AE 6394 coursework and was extended
independently afterward. The later work focused on a reproducible experiment
contract, matched architecture comparisons, force checks, and clear boundaries
around evidence that is still pending.

The current measured scope is interpolation on the official public AirfRANS
data. The repository records OOD, scarce-data, uncertainty, and active-learning
studies as separate work rather than presenting them as completed evidence.

## Success criteria

- Split simulations, not individual mesh nodes, so a CFD solution cannot appear
  in both training and testing.
- Normalize from training data only and use the same evaluation cases and seeds
  for every model family.
- Report field metrics alongside lift and drag error.
- Preserve the configuration, checkpoint identity, and per-case results needed
  to reproduce a reported comparison.
- Treat OOD behaviour and uncertainty calibration as required evidence before
  any operational-use claim.

## Non-goals

AIRFAANS is not a replacement for validated CFD, an airworthiness model, or a
production aircraft-design system. Its current purpose is a reproducible
scientific-ML comparison and a clear account of what the completed experiments
do and do not establish.
