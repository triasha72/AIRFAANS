#!/usr/bin/env python3
"""Run the reproducible AIRFAANS training/evaluation matrix.

This command deliberately refuses to invent evidence. It either runs the
requested official-data treatments or emits a dry-run plan that can be copied
to a GPU host. Every treatment resumes from its own checkpoint directory.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

TASKS = ("scarce", "reynolds_ood", "aoa_ood")
SEEDS = (17, 29, 41)


def command(args: argparse.Namespace, task: str, seed: int) -> list[str]:
    output = args.output_root / task / f"{args.model}-seed{seed}"
    argv = [
        sys.executable,
        "-m",
        "airfaans.cli",
        "train",
        "--dataset-root",
        str(args.dataset_root),
        "--manifest",
        str(args.manifest),
        "--config",
        str(args.config),
        "--model",
        args.model,
        "--task",
        task,
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
    ]
    if any((output / name).is_file() for name in ("latest.pt", "best.pt")):
        argv.append("--resume")
    return argv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/airfrans_tasks_v0_1.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_v0_1.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/evidence_matrix"))
    parser.add_argument("--model", default="mesh_graph_net")
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset_root.exists():
        parser.error(f"AirfRANS dataset root does not exist: {args.dataset_root}")
    if not args.manifest.exists():
        parser.error(f"task manifest does not exist: {args.manifest}")
    if not args.config.exists():
        parser.error(f"experiment config does not exist: {args.config}")

    tasks = tuple(args.task or TASKS)
    seeds = tuple(args.seed or SEEDS)
    commands = [command(args, task, seed) for task in tasks for seed in seeds]
    for argv in commands:
        print(shlex.join(argv))
        if not args.dry_run:
            subprocess.run(argv, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
