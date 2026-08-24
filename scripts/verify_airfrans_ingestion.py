"""Freeze evidence that a real AirfRANS VTK case traverses the graph pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from airfaans.airfrans import load_graph


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case_id")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/airfrans_ingestion_v0_1.json")
    )
    args = parser.parse_args()
    started = perf_counter()
    graph = load_graph(args.root / args.case_id)
    elapsed = perf_counter() - started
    flow = graph.flow
    payload = {
        "evidence_label": "measured_real_airfrans_ingestion",
        "case_id": flow.case_id,
        "source_files": {
            suffix: sha256(args.root / flow.case_id / f"{flow.case_id}_{suffix}")
            for suffix in ("internal.vtu", "aerofoil.vtp", "freestream.vtp")
        },
        "node_count": len(flow.points),
        "surface_node_count": int(flow.surface_mask.sum()),
        "directed_edge_count": graph.edge_index.shape[1],
        "node_feature_count": flow.features.shape[1],
        "edge_feature_count": graph.edge_features.shape[1],
        "target_field_count": flow.targets.shape[1],
        "reynolds": flow.reynolds,
        "angle_of_attack_deg": flow.angle_of_attack_deg,
        "target_ranges": {
            name: [float(np.min(flow.targets[:, index])), float(np.max(flow.targets[:, index]))]
            for index, name in enumerate(
                ("velocity_x", "velocity_y", "pressure", "turbulent_viscosity")
            )
        },
        "ingestion_elapsed_seconds": elapsed,
        "timing_scope": "local Apple Silicon VTK read, validation, normal mapping, graph build",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
