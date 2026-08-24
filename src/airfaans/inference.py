"""Validated checkpoint loading and case-level inference."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from airfaans.airfrans import load_case
from airfaans.experiment import (
    ExperimentConfig,
    _case_tensors,
    _forward,
    build_model,
    case_directory,
    sample_indices,
)
from airfaans.normalization import Normalization


class CheckpointPredictor:
    """Load one experiment checkpoint and predict an indexed AirfRANS case."""

    def __init__(self, checkpoint_path: Path, dataset_root: Path, device: str = "cpu") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_root = Path(dataset_root)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.config = ExperimentConfig(**payload["config"])
        self.normalization = Normalization.from_dict(payload["normalization"])
        self.model = build_model(self.config, len(self.normalization.feature_mean)).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    def __call__(self, request) -> dict[str, object]:
        started = perf_counter()
        case = load_case(case_directory(self.dataset_root, request.case_id))
        # Conditions are part of case identity. Refuse mismatched metadata instead
        # of silently claiming counterfactual geometry inference.
        if not np.isclose(case.reynolds, request.reynolds, rtol=1e-5):
            raise ValueError("request Reynolds number does not match the indexed case")
        if not np.isclose(case.angle_of_attack_deg, request.angle_of_attack_deg, atol=1e-5):
            raise ValueError("request angle of attack does not match the indexed case")
        indices = sample_indices(case, self.config.nodes_per_case, self.config.seed)
        x, _, edges, edge_features = _case_tensors(
            case, self.normalization, indices, self.device, self.config.model
        )
        with self.torch.inference_mode():
            normalized = _forward(self.model, self.config.model, x, edges, edge_features)
        prediction = self.normalization.inverse_targets(normalized.cpu().numpy())
        return {
            "model_id": f"{self.config.model}-seed-{self.config.seed}",
            "node_count": len(indices),
            "field_means": {
                name: float(value)
                for name, value in zip(
                    ("velocity_x", "velocity_y", "pressure", "turbulent_viscosity"),
                    prediction.mean(axis=0),
                    strict=True,
                )
            },
            "lift_coefficient": None,
            "drag_coefficient": None,
            "mean_uncertainty": None,
            "model_latency_ms": (perf_counter() - started) * 1000.0,
            "evidence_label": "checkpoint_inference_pressure_only_forces_not_verified",
        }
