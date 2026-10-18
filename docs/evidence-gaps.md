# AIRFAANS evidence audit — 2026-10-18

This audit compares the published GitHub baseline (`60f487a`), the prior
AirFAANS and portfolio-audit chats, the older local checkout, CI history,
project-related GitHub notification emails, and the connected Drive folder.
Chat descriptions record prior work; they do not replace inspectable artifacts.

## Confirmed evidence

The checked-in interpolation comparison contains three architectures, seeds
17/29/41, and 200 full-mesh test cases per treatment. MeshGraphNet has the
lowest mean error across all four fields and drag; PNO has the lowest mean lift
error. The repository also preserves real ingestion, optimization-smoke and
force-convention checks. These are distinct evidence categories.

The README and chat history report six Reynolds-OOD treatments (three models,
seeds 29/41, 496 cases each) and two scarce-data MLP treatments (seeds 17/23,
200 cases each). Their raw bundles and compact metric summaries are absent from
the published tree. The connected `AIRFAANS/results` folder returned no files.
The previously used Colab notebook requires access from its original account.
This does not establish that the artifacts are lost; it prevents independent
revalidation in this audit. Do not silently convert those historical reports
into new verified measurements.

## Repairs in this change

| Gap | Repair and verification |
|---|---|
| CI fails during test collection | Declare repository/source paths in pytest configuration so the documented bare `pytest` command can import `scripts` and `airfaans`. |
| OOD threshold differs from the roadmap | Use `>= 1.1` consistently; test the boundary and reject bounded or incorrectly paired tasks. |
| Invalid uncertainty values can pass or crash checks | Reject non-finite/negative uncertainty and invalid correlations; readiness fails closed on invalid numeric values and repeated seeds. |
| Readiness cannot consume evaluator output | Read nested ensemble summaries and same-checkpoint OOD comparison reports; reject mismatched comparison hashes. |
| Older ensemble function samples meshes but reports unbounded results | Evaluate all nodes; test real checkpoint loading and inference on a small synthetic fixture. The fixture is software evidence only. |
| README examples use obsolete CLI flags | Use per-task output directories, aggregate exact coverage, then compare saved reports. |
| Published 3-D claim and missing link | Restore an accurate extension document and describe the published implementation as 2-D. |
| Stale model ranking and matrix scope | Use the completed three-seed ranking and explain nine default treatments versus 27 across all models. |

## Remaining work and closure criteria

| Priority | Gap | What closes it |
|---|---|---|
| 1 | Recover durable artifacts | Locate the original Drive account/folder; inventory the six Reynolds and two scarce bundles; verify checkpoint/bundle hashes, case IDs, exact coverage and finite metrics; publish compact summaries with durable raw-artifact references. |
| 2 | Complete the matched task matrix | Reconcile recovered treatments with seeds 17/29/41 for all models. Seed 23 is additional scarce replication, not a substitute for the matched seed contract. Evaluate AoA-OOD on all 196 official meshes. |
| 3 | Measure ensemble uncertainty | Use one fixed training-task checkpoint ensemble on interpolation (200), Reynolds-OOD (496) and AoA-OOD (196). Preserve full-mesh records, calibration diagnostics, correlation and same-checkpoint comparisons. A two-member ensemble remains exploratory. |
| 4 | Complete active learning | Implement protected candidate-pool checks, identical initial sets/budgets, per-round retraining receipts, and independently repeated random/uncertainty arms. Selection functions alone do not establish improvement. |
| 5 | Measure scaling and performance | Record hardware, package versions, mesh sizes, peak memory, training throughput and inference latency. DDP primitives exist; FSDP and measured multi-GPU scaling do not. |
| 6 | Extend physical structure | Define and test frame-consistency separately from Galilean invariance; follow the 3-D extension criteria before claiming 3-D CFD capability. |

The current operational-readiness decision must remain rejected. Code repairs
and synthetic regression fixtures cannot replace the missing experiments.

## Recovery and execution

1. Use the original Google account to inspect the prior Colab recovery and
   experiment notebooks. History identifies Pointwise MLP seeds 29/41 as the next
   ensemble candidates; verify their actual paths and hashes before evaluation.
2. Keep compact checkpoints and per-case records on persistent storage. Download
   and verify the official dataset on the compute host; history records a corrupt
   archived download, so an old ZIP filename is not sufficient proof of integrity.
3. Inspect the [matrix commands](../ROADMAP.md#reproducible-execution) with
   `--dry-run`, reconcile completed evidence, then execute only missing treatments.
4. Follow the [restart-safe runbook](real-data-runbook.md#restart-safe-ensemble-evaluation).
   Copy each final `ensemble_result.json` and `compare-ood` output to the evidence
   directory with distinct filenames before running the readiness assessment.
5. Publish measured results after raw records and their provenance are verified.

The older checkout has uncommitted changes and divergent history. Preserve it;
review and port its changes individually rather than resetting it or assuming
they are already present on GitHub. PR #1 also remains open even though main
contains related UQ functionality; compare its current diff before integrating it.

## Sources

- [Published repository](https://github.com/triasha72/AIRFAANS/tree/60f487a47f1b770834e3945accdc53f2bca09c33)
- [Failing CI baseline](https://github.com/triasha72/AIRFAANS/actions/runs/37206889066)
- [Existing UQ PR #1](https://github.com/triasha72/AIRFAANS/pull/1)
- [Checked-in readiness decision](../artifacts/evaluation/operational_readiness_v1.json)
- [Frozen interpolation summary](../artifacts/evaluation/interpolation_three_seed_summary.json)

Private chat history and connected Drive reads informed the recovery status;
they are not public raw-result citations.
