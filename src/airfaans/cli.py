"""AIRFAANS command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from airfaans.evaluation import field_metrics
from airfaans.graph import knn_graph
from airfaans.io import save_case
from airfaans.physics import integrate_pressure_forces
from airfaans.synthetic import make_case


def demo(output: Path) -> dict[str, object]:
    case = make_case()
    graph = knn_graph(case, neighbors=8)
    force = integrate_pressure_forces(case, case.targets)
    payload = {
        "evidence_label": "analytic_fixture_not_airfrans",
        "case_id": case.case_id,
        "node_count": len(case.points),
        "directed_edge_count": graph.edge_index.shape[1],
        "feature_count": case.features.shape[1],
        "target_fields": ["velocity_x", "velocity_y", "pressure", "turbulent_viscosity"],
        "identity_metrics": field_metrics(case.targets, case.targets),
        "integrated_coefficients": {"lift": force.lift, "drag": force.drag},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_case(output.with_suffix(".npz"), case)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="airfaans")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="Run the deterministic analytic fixture.")
    demo_parser.add_argument(
        "--output", type=Path, default=Path("artifacts/local/synthetic_demo.json")
    )
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved case.")
    inspect_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(demo(args.output), indent=2))
    elif args.command == "inspect":
        payload = np.load(args.path, allow_pickle=False)
        print({name: payload[name].shape for name in payload.files})


if __name__ == "__main__":
    main()
