"""AIRFAANS command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from airfaans.evaluation import field_metrics
from airfaans.experiment import (
    MODEL_NAMES,
    TASK_SPLITS,
    aggregate_evaluation_shards,
    config_from_yaml,
    evaluate_checkpoint_shard,
    run_experiment,
)
from airfaans.graph import knn_graph
from airfaans.io import save_case
from airfaans.physics import integrate_pressure_forces
from airfaans.release import validate_release
from airfaans.synthetic import make_case


def training_summary(result: dict[str, object]) -> dict[str, object] | None:
    """Return the user-facing rank-zero summary, or nothing for DDP workers."""
    if result.get("evidence_label") == "distributed_worker_complete":
        return None
    return {
        key: result[key]
        for key in (
            "evidence_label",
            "device",
            "best_validation_mean_relative_l2",
            "checkpoint",
        )
    }


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
    train_parser = subparsers.add_parser("train", help="Run an AirfRANS treatment.")
    train_parser.add_argument("--dataset-root", type=Path, required=True)
    train_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/airfrans_tasks_v0_1.json")
    )
    train_parser.add_argument("--config", type=Path, default=Path("configs/experiment_v0_1.yaml"))
    train_parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    train_parser.add_argument("--task", choices=tuple(TASK_SPLITS), default="interpolation")
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--max-train-cases", type=int)
    train_parser.add_argument("--max-validation-cases", type=int)
    train_parser.add_argument("--max-test-cases", type=int)
    train_parser.add_argument("--epochs", type=int, help="Explicit bounded-run override.")
    train_parser.add_argument("--nodes-per-case", type=int, help="Explicit bounded-run override.")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--strategy", choices=("single", "ddp"), default="single")
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run a resumable full-mesh official-test shard."
    )
    evaluate_parser.add_argument("--dataset-root", type=Path, required=True)
    evaluate_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/airfrans_tasks_v0_1.json")
    )
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--start", type=int, default=0)
    evaluate_parser.add_argument("--count", type=int)
    evaluate_parser.add_argument("--no-resume", action="store_true")
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Validate all official-test shards and write result.json."
    )
    aggregate_parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/airfrans_tasks_v0_1.json")
    )
    aggregate_parser.add_argument("--checkpoint", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    release_parser = subparsers.add_parser(
        "validate-release", help="Verify checkpoint, evaluation, and dataset release evidence."
    )
    release_parser.add_argument("--release-manifest", type=Path, required=True)
    release_parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(demo(args.output), indent=2))
    elif args.command == "inspect":
        payload = np.load(args.path, allow_pickle=False)
        print({name: payload[name].shape for name in payload.files})
    elif args.command == "train":
        from dataclasses import replace

        config = config_from_yaml(args.config, args.model, args.task, args.seed)
        if args.epochs is not None:
            config = replace(config, epochs=args.epochs)
        if args.nodes_per_case is not None:
            config = replace(config, nodes_per_case=args.nodes_per_case)
        result = run_experiment(
            args.dataset_root,
            args.manifest,
            args.output_dir,
            config,
            args.max_train_cases,
            args.max_validation_cases,
            args.max_test_cases,
            args.resume,
            args.strategy,
        )
        summary = training_summary(result)
        if summary is not None:
            print(json.dumps(summary, indent=2))
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate_checkpoint_shard(
                    args.dataset_root,
                    args.manifest,
                    args.checkpoint,
                    args.output_dir,
                    args.start,
                    args.count,
                    not args.no_resume,
                ),
                indent=2,
            )
        )
    elif args.command == "aggregate":
        print(
            json.dumps(
                aggregate_evaluation_shards(args.manifest, args.checkpoint, args.output_dir),
                indent=2,
            )
        )
    elif args.command == "validate-release":
        print(
            json.dumps(validate_release(args.release_manifest, args.checkpoint).__dict__, indent=2)
        )


if __name__ == "__main__":
    main()
