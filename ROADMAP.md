# AIRFAANS: the remaining research work

[Read the project overview and measured results](README.md)

The interpolation study is complete, including three architectures, three
seeds, all 200 official test meshes, and force-coefficient checks. It also made
one point clear: no architecture wins on every output. MeshGraphNet produced the
best pressure and drag errors at this budget, while the pointwise MLP did better
on several other quantities.

## The question to answer next

Can uncertainty tell us when the surrogate is leaving familiar aerodynamic
conditions?

The code for checkpoint ensembles, per-case uncertainty, OOD comparison, and
uncertainty-guided case selection is ready. The missing part is the expensive
run on the official Reynolds-number and angle-of-attack OOD tasks. Until those
runs exist, the repository makes no claim that its uncertainty estimate is
useful.

## Planned run order

1. Evaluate the same frozen ensemble on interpolation, Reynolds OOD, and angle-
   of-attack OOD cases.
2. Compare uncertainty with actual field and lift/drag error for every case.
3. Repeat simulation selection with uncertainty and random acquisition using
   matched budgets and seeds.
4. Profile accuracy, memory, and inference time at useful mesh sizes.
5. Test another geometry family before discussing broad generalization.

A credible next release needs field metrics, engineering force metrics, seed
variation, and a useful relationship between uncertainty and error. A better
interpolation average by itself will not close this gap.
