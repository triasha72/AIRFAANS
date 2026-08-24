"""Render real AirfRANS reference fields for a selected simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from airfaans.airfrans import load_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = load_case(args.root / args.case_id)
    fields = (
        (case.targets[:, 2], "Pressure / density", "coolwarm"),
        (np.linalg.norm(case.targets[:, :2], axis=1), "Velocity magnitude", "viridis"),
        (case.targets[:, 3], "Turbulent viscosity", "magma"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    for axis, (values, label, palette) in zip(axes, fields, strict=True):
        plot = axis.scatter(case.points[:, 0], case.points[:, 1], c=values, s=0.15, cmap=palette)
        axis.set(xlim=(-0.5, 2.0), ylim=(-0.75, 0.75), xlabel="x / chord", ylabel="y / chord")
        axis.set_aspect("equal")
        axis.set_title(label, loc="left", fontsize=10)
        figure.colorbar(plot, ax=axis, fraction=0.025, pad=0.02)
    figure.suptitle(
        f"AirfRANS reference fields | Re={case.reynolds / 1e6:.3f}M | "
        f"AoA={case.angle_of_attack_deg:.3f}°",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    metadata = {
        "case_id": case.case_id,
        "evidence_label": "real_airfrans_reference_not_model_prediction",
        "node_count": len(case.points),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
