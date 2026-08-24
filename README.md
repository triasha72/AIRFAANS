# AIRFAANS

Geometry-aware scientific machine learning for aerodynamic CFD fields.

**Coursework project for AE 6394 at the Georgia Institute of Technology.**
The repository documents the implementation and evidence produced for the
project; it does not imply endorsement by Georgia Tech or the AirfRANS authors.

AIRFAANS takes an airfoil mesh, freestream condition, Reynolds number, and angle
of attack and predicts four values at every mesh node:

```text
(geometry, Re, angle of attack) -> (velocity_x, velocity_y, pressure, turbulent viscosity)
```

The project compares a pointwise MLP, a residual MeshGraphNet-style model, and a
global point operator. The evaluation contract covers interpolation, scarce-data
learning, Reynolds-number extrapolation, and angle-of-attack extrapolation. It
also checks whether predicted fields preserve pressure coefficient and integrated
lift and drag, whether ensemble uncertainty rises out of distribution, and
whether uncertainty-guided simulation selection beats random selection.

AIRFAANS is an independent project, not part of or endorsed by the AirfRANS
authors. AirfRANS data and model weights are not redistributed here.

## Current evidence

| Capability | Status | Evidence |
|---|---|---|
| Typed mesh/field representation | Implemented and tested | `src/airfaans/data.py` |
| Mesh-connectivity and k-NN graphs | Implemented and tested | `src/airfaans/graph.py` |
| Pointwise MLP | Implemented | `src/airfaans/models.py` |
| MeshGraphNet-style message passing | Implemented | `src/airfaans/models.py` |
| Irregular-point neural operator | Implemented | `src/airfaans/models.py` |
| Official VTK/PyVista ingestion | Measured on a real 181,794-node case | `artifacts/evaluation/airfrans_ingestion_v0_1.json` |
| Official task manifest | Frozen and validated for all 1,000 cases | `data/manifests/airfrans_tasks_v0_1.json` |
| Real-data optimization smoke | All three models reduced loss by 64–88% | `artifacts/evaluation/real_tiny_overfit_v0_1.json` |
| AirfRANS interpolation/OOD measurements | Pending dataset/GPU execution | `reports/airfrans_v0_1.md` |
| Ensemble UQ and active learning | Metrics/selection implemented; experiment pending | tests and config |
| FastAPI and Docker | Implemented; checkpoint required for inference | `/health`, `/v1/predict` |

The checked-in demo uses a deterministic analytic cylinder-like fixture. It
tests the pipeline in CI, but it is neither RANS CFD nor an AirfRANS result.

The real-data ingestion path has also been executed locally. One official
AirfRANS training simulation produced 181,794 nodes, 1,025 surface nodes, and a
724,640-edge mesh graph in 1.57 seconds on Apple Silicon. This is measured data
engineering evidence, not model-accuracy evidence.

A bounded 512-node CPU optimization check also passed for all three model
families. Over 100 steps, normalized training MSE fell by 64.4% for the MLP,
87.7% for MeshGraphNet, and 67.5% for the point operator. This confirms that the
real-data tensors, graph, gradients, and optimizers connect correctly; it is not
a held-out comparison and should not be presented as model quality.

![Real AirfRANS pressure, velocity and turbulent-viscosity reference fields](docs/assets/airfrans_reference_case_v0_1.png)

## Why AirfRANS

The [official AirfRANS library](https://github.com/Extrality/airfrans_lib)
provides 1,000 incompressible RANS simulations over NACA 4- and 5-digit
airfoils, with Reynolds numbers from 2 to 6 million and angles of attack from
-5 to 15 degrees. Its four tasks are `full`, `scarce`, `reynolds`, and `aoa`.
The documented nodal inputs include position, inlet velocity, signed distance,
surface membership, and normals; targets include velocity, pressure, and
turbulent viscosity. See the
[AirfRANS simulation reference](https://airfrans.readthedocs.io/en/latest/notes/simulation.html).

Graphs are useful because the CFD domain is irregular and resolution varies near
the airfoil. AIRFAANS represents each mesh node as a graph node, using either
solver connectivity or a controlled k-NN graph. Edge features are relative
coordinates and distance. The processor follows the encode-process-decode idea
introduced by [MeshGraphNets](https://arxiv.org/abs/2010.03409), while keeping
the implementation small enough to audit. PyTorch Geometric conversion is
provided for graph batching and interoperability with its documented
[message-passing interface](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html).

## Architecture

```text
AirfRANS / OpenFOAM VTK
          |
          v
PyVista field and mesh extraction -- Dask case-level preprocessing
          |
          v
typed FlowCase -> mesh or k-NN graph -> PyTorch Geometric Data
          |
          +---------+------------------+
          |         |                  |
          v         v                  v
   pointwise MLP  MeshGraphNet   point neural operator
          |         |                  |
          +---------+------------------+
                    |
                    v
     velocity, pressure, turbulent-viscosity fields
                    |
                    v
  field metrics | Cp/CL/CD | ensembles | OOD | active learning
                    |
                    v
           MLflow artifacts + FastAPI
```

## Five-minute start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
airfaans demo
pytest -q
```

The demo writes `artifacts/local/synthetic_demo.json` and a compressed flow case.
Its `evidence_label` is deliberately `analytic_fixture_not_airfrans`.

## Real AirfRANS setup

Install the complete environment:

```bash
python -m pip install -e ".[ml,airfrans,tracking,api,dev]"
python -m pip install airfrans
```

Download AirfRANS using its official package:

```python
import airfrans as af

af.dataset.download(root="data/raw/airfrans", unzip=True)
```

Do not commit the dataset. The downloaded processed archive was verified as
10,029,067,577 bytes with SHA-256
`6b301d75dee77fc6c7de6e551c44332be4acc84e30b253e9c60cfa756f6c96db`.
The checked-in manifest preserves this identity and the official task splits.
The AirfRANS adapter maps the actual `U`, `p`, `nut`, `implicit_distance`, and
`Normals` arrays and fails on missing fields rather than silently guessing.

## Experiments

The preregistration is `configs/experiment_v0_1.yaml`. Its comparison keeps
data, normalization, optimization, seeds, and evaluation cases fixed across:

1. Pointwise MLP: local node and operating-condition features, no connectivity.
2. MeshGraphNet: residual edge-to-node message passing over mesh or k-NN edges.
3. Point operator: global learned modes over the irregular point set.

Splits are made by simulation, never by randomly mixing nodes from one CFD case
across train and test. Reynolds and angle-of-attack OOD tasks hold out the upper
operating range. Results must include three seeds, per-field RMSE/MAE/relative
L2, CL/CD error, latency, memory, node/edge throughput, uncertainty-error
correlation, and the OOD/ID uncertainty ratio.

The complete manual run and publication gates are in
[`docs/real-data-runbook.md`](docs/real-data-runbook.md). Pending tables are in
[`reports/airfrans_v0_1.md`](reports/airfrans_v0_1.md).

## API

```bash
python -m pip install -e ".[api]"
uvicorn airfaans.api:app --reload
curl http://localhost:8000/health
```

`/v1/predict` returns HTTP 503 until a trained, validated checkpoint is wired to
the predictor boundary. That fail-closed behavior prevents the analytic fixture
or an unvalidated model from appearing as an engineering result.

## Scaling path

- Dask parallelizes independent VTK preprocessing tasks.
- PyG conversion supports graph mini-batching.
- Training supports CUDA AMP, gradient accumulation, checkpointing, and resume.
- The real run records node/edge counts, peak accelerator memory, training
  throughput, and per-case inference latency.
- DDP is a measured optional treatment, not a default requirement.

## Repository map

```text
configs/       frozen experiment contracts
docs/          architecture and real-data runbook
reports/       pending and measured experiment reports
src/airfaans/  data, graph, model, physics, UQ, training, API code
tests/         deterministic unit and integration tests
```

## Scope and limitations

- The analytic fixture exists for CI and API/data-contract development only.
- The point operator is a compact irregular-domain treatment, not a claim of
  reproducing FNO, GINO, or Transolver.
- Pressure-only force integration is a first engineering check; a publication
  result should add the AirfRANS/OpenFOAM surface convention and viscous shear.
- Inference speedup versus CFD remains pending because solver wall time and model
  inference must be measured on named hardware.
- This work is identified as an AE 6394 coursework project. Georgia Tech and the
  AirfRANS authors do not maintain or endorse this repository.

## License

Code is MIT licensed. AirfRANS and OpenFOAM data remain governed by their own
licenses and source terms.
