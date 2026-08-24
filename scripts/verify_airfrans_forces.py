"""Freeze reference force coefficients for real AirfRANS cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from airfaans.experiment import case_directory
from airfaans.physics import integrate_airfrans_forces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/airfrans_tasks_v0_1.json")
    )
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/airfrans_force_verification_v0_1.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case_ids = manifest["splits"]["full_train"][: args.cases]
    results = {
        case_id: asdict(integrate_airfrans_forces(case_directory(args.dataset_root, case_id)))
        for case_id in case_ids
    }
    payload = {
        "evidence_label": "official_airfrans_reference_field_force_convention",
        "case_count": len(results),
        "includes_pressure_and_molecular_viscous_contributions": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
