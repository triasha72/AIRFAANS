"""Run a bounded optimization check for all models on one real AirfRANS case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from airfaans.airfrans import load_case
from airfaans.models import mesh_graph_net, parameter_count, point_neural_operator, pointwise_mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/real_tiny_overfit_v0_1.json")
    )
    args = parser.parse_args()
    import torch

    torch.manual_seed(args.seed)
    case = load_case(args.root / args.case_id)
    generator = np.random.default_rng(args.seed)
    surface = np.flatnonzero(case.surface_mask)
    volume = np.flatnonzero(~case.surface_mask)
    surface_count = min(len(surface), max(32, args.nodes // 8))
    indices = np.concatenate(
        (
            generator.choice(surface, size=surface_count, replace=False),
            generator.choice(volume, size=args.nodes - surface_count, replace=False),
        )
    )
    indices.sort()
    features = case.features[indices]
    targets = case.targets[indices]
    feature_mean, feature_std = features.mean(0), features.std(0)
    target_mean, target_std = targets.mean(0), targets.std(0)
    x = torch.from_numpy((features - feature_mean) / np.maximum(feature_std, 1e-6)).float()
    y = torch.from_numpy((targets - target_mean) / np.maximum(target_std, 1e-6)).float()
    distance = np.sum((case.points[indices, None] - case.points[None, indices]) ** 2, axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :8]
    source = np.repeat(np.arange(args.nodes), 8)
    target = nearest.reshape(-1)
    edge_index = torch.from_numpy(np.vstack((source, target))).long()
    relative = case.points[indices][target] - case.points[indices][source]
    edge_features = torch.from_numpy(
        np.column_stack((relative, np.linalg.norm(relative, axis=1)))
    ).float()
    treatments = {
        "pointwise_mlp": (
            pointwise_mlp(x.shape[1], hidden_dim=64),
            lambda model: model(x),
        ),
        "mesh_graph_net": (
            mesh_graph_net(x.shape[1], hidden_dim=64, layers=3),
            lambda model: model(x, edge_index, edge_features),
        ),
        "point_neural_operator": (
            point_neural_operator(x.shape[1], hidden_dim=64, modes=16),
            lambda model: model(x),
        ),
    }
    results = {}
    for name, (model, forward) in treatments.items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        started = perf_counter()
        for _ in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(forward(model), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        results[name] = {
            "parameters": parameter_count(model),
            "initial_normalized_mse": losses[0],
            "final_normalized_mse": losses[-1],
            "loss_reduction_fraction": 1.0 - losses[-1] / losses[0],
            "elapsed_seconds": perf_counter() - started,
            "passed": losses[-1] < losses[0] * 0.5,
        }
    payload = {
        "evidence_label": "real_airfrans_bounded_optimization_check_not_benchmark",
        "case_id": args.case_id,
        "sampled_nodes": args.nodes,
        "surface_nodes": surface_count,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": "cpu",
        "treatments": results,
        "all_passed": all(result["passed"] for result in results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"]:
        raise SystemExit("At least one bounded optimization check failed")


if __name__ == "__main__":
    main()
