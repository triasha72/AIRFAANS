"""Validate and freeze the official AirfRANS task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from airfaans.airfrans import parse_condition


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/airfrans_tasks_v0_1.json")
    )
    args = parser.parse_args()
    source = args.root / "manifest.json"
    official = json.loads(source.read_text(encoding="utf-8"))
    expected = {
        "full_train": 800,
        "scarce_train": 200,
        "reynolds_train": 504,
        "aoa_train": 804,
        "full_test": 200,
        "reynolds_test": 496,
        "aoa_test": 196,
    }
    counts = {name: len(values) for name, values in official.items()}
    if counts != expected:
        raise ValueError(f"Unexpected official split counts: {counts}")
    all_cases = sorted(set().union(*map(set, official.values())))
    missing = [case_id for case_id in all_cases if not (args.root / case_id).is_dir()]
    if missing:
        raise ValueError(f"Missing {len(missing)} case directories")
    conditions = {case_id: parse_condition(case_id) for case_id in all_cases}
    payload = {
        "source": "AirfRANS processed Dataset.zip",
        "source_manifest_sha256": sha256(source),
        "dataset_archive_sha256": (
            (args.root.parent.parent / "Dataset.zip.sha256").read_text().split()[0]
        ),
        "case_count": len(all_cases),
        "file_count": sum(1 for path in args.root.rglob("*") if path.is_file()),
        "split_counts": counts,
        "condition_ranges": {
            "inlet_speed": [
                min(item.inlet_speed for item in conditions.values()),
                max(item.inlet_speed for item in conditions.values()),
            ],
            "angle_of_attack_deg": [
                min(item.angle_of_attack_deg for item in conditions.values()),
                max(item.angle_of_attack_deg for item in conditions.values()),
            ],
            "reynolds": [
                min(item.reynolds for item in conditions.values()),
                max(item.reynolds for item in conditions.values()),
            ],
        },
        "splits": official,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
