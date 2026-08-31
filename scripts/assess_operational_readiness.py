#!/usr/bin/env python3
"""Build a deterministic AIRFAANS evidence-completeness decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airfaans.readiness import assess_readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.evidence_directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            summaries.append(payload)
    result = assess_readiness(summaries)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
