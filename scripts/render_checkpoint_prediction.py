"""Render CFD, checkpoint prediction, and absolute error on sampled real nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from airfaans.airfrans import load_case
from airfaans.experiment import _case_tensors, _forward, case_directory, sample_indices
from airfaans.inference import CheckpointPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    predictor = CheckpointPredictor(args.checkpoint, args.dataset_root)
    case = load_case(case_directory(args.dataset_root, args.case_id))
    indices = sample_indices(case, predictor.config.nodes_per_case, predictor.config.seed)
    x, _, edges, edge_features = _case_tensors(
        case, predictor.normalization, indices, predictor.device, predictor.config.model
    )
    with predictor.torch.inference_mode():
        normalized = _forward(predictor.model, predictor.config.model, x, edges, edge_features)
    prediction = predictor.normalization.inverse_targets(normalized.cpu().numpy())
    truth = case.targets[indices]
    points = case.points[indices]
    values = (
        (
            np.linalg.norm(truth[:, :2], axis=1),
            np.linalg.norm(prediction[:, :2], axis=1),
            "Velocity magnitude",
        ),
        (truth[:, 2], prediction[:, 2], "Pressure"),
        (truth[:, 3], prediction[:, 3], "Turbulent viscosity"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for row, (reference, estimate, label) in enumerate(values):
        lower = min(reference.min(), estimate.min())
        upper = max(reference.max(), estimate.max())
        error = np.abs(estimate - reference)
        for column, (field, title) in enumerate(
            ((reference, "CFD"), (estimate, "Prediction"), (error, "Absolute error"))
        ):
            limits = (0.0, error.max()) if column == 2 else (lower, upper)
            artist = axes[row, column].scatter(
                points[:, 0],
                points[:, 1],
                c=field,
                s=5,
                cmap="viridis",
                vmin=limits[0],
                vmax=limits[1],
            )
            axes[row, column].set_title(f"{label}: {title}")
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("x / chord")
            axes[row, column].set_ylabel("y / chord")
            figure.colorbar(artist, ax=axes[row, column], shrink=0.8)
    figure.suptitle("Bounded integration run — not an AirfRANS benchmark", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
