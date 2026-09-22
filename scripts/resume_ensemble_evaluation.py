#!/usr/bin/env python3
"""Plan the next safe shard of a persisted ensemble evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airfaans.uq_experiment import (
    EnsembleManifest,
    audit_ensemble_records,
    load_ensemble_manifest,
    write_json_atomic,
)


def plan_next_shard(
    records_directory: Path,
    manifest: EnsembleManifest,
    evaluation_task: str,
    expected_cases: int,
) -> dict[str, object]:
    audit = audit_ensemble_records(
        records_directory, manifest, evaluation_task, expected_cases
    )
    state = {
        "saved_cases": audit.saved_cases,
        "expected_cases": audit.expected_cases,
        "next_missing_index": audit.next_missing_index,
        "checkpoint_sha256": list(manifest.checkpoint_sha256),
    }
    write_json_atomic(records_directory.parent / "resume_state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--evaluation-task", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()
    manifest = load_ensemble_manifest(args.checkpoint)
    state = plan_next_shard(
        args.output_dir / "ensemble_cases",
        manifest,
        args.evaluation_task,
        args.expected_cases,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
