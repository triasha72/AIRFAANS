"""Executable multi-case AirfRANS training and evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from airfaans.airfrans import load_case, load_graph
from airfaans.distributed import context_from_environment, partition_equal
from airfaans.evaluation import field_metrics
from airfaans.models import mesh_graph_net, parameter_count, point_neural_operator, pointwise_mlp
from airfaans.normalization import Normalization, RunningMoments
from airfaans.physics import integrate_airfrans_forces

MODEL_NAMES = ("pointwise_mlp", "mesh_graph_net", "point_neural_operator")
TASK_SPLITS = {
    "interpolation": ("full_train", "full_test"),
    "scarce": ("scarce_train", "full_test"),
    "reynolds_ood": ("reynolds_train", "reynolds_test"),
    "aoa_ood": ("aoa_train", "aoa_test"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    task: str = "interpolation"
    seed: int = 17
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 128
    processor_layers: int = 6
    modes: int = 32
    nodes_per_case: int = 4096
    cases_per_epoch: int = 32
    validation_cases: int = 20
    patience: int = 20
    minimum_delta: float = 1e-5
    mixed_precision: bool = True

    def validate(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"model must be one of {MODEL_NAMES}")
        if self.task not in TASK_SPLITS:
            raise ValueError(f"task must be one of {tuple(TASK_SPLITS)}")
        for name in ("epochs", "nodes_per_case", "cases_per_epoch", "validation_cases"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def config_from_yaml(path: Path, model: str, task: str, seed: int) -> ExperimentConfig:
    """Resolve one treatment from the frozen comparison configuration."""
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    training = payload["training"]
    model_config = payload["models"][model]
    config = ExperimentConfig(
        model=model,
        task=task,
        seed=seed,
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        hidden_dim=int(model_config["hidden_dim"]),
        processor_layers=int(model_config.get("processor_layers", 6)),
        modes=int(model_config.get("modes", 32)),
        nodes_per_case=int(training.get("nodes_per_case", 4096)),
        cases_per_epoch=int(training.get("cases_per_epoch", 32)),
        validation_cases=int(training.get("validation_cases", 20)),
        patience=int(training.get("early_stopping_patience", 20)),
        mixed_precision=bool(training.get("mixed_precision", True)),
    )
    config.validate()
    return config


def official_split(
    manifest_path: Path, task: str, validation_cases: int
) -> tuple[list[str], list[str], list[str]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    train_name, test_name = TASK_SPLITS[task]
    train_pool = list(payload["splits"][train_name])
    test = list(payload["splits"][test_name])
    if len(train_pool) <= validation_cases:
        raise ValueError("training split is too small for the requested validation set")
    # The official test set remains untouched; validation is a deterministic tail
    # of the official training list and is recorded in every artifact.
    return train_pool[:-validation_cases], train_pool[-validation_cases:], test


def case_directory(dataset_root: Path, case_id: str) -> Path:
    direct = Path(dataset_root) / case_id
    if direct.is_dir():
        return direct
    nested = Path(dataset_root) / "Dataset" / case_id
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"AirfRANS case not found: {case_id}")


def sample_indices(case, count: int, seed: int) -> np.ndarray:
    """Deterministically retain boundary nodes while sampling the volume."""
    if count >= len(case.points):
        return np.arange(len(case.points))
    generator = np.random.default_rng(seed)
    surface = np.flatnonzero(case.surface_mask)
    volume = np.flatnonzero(~case.surface_mask)
    surface_count = min(len(surface), max(1, count // 8))
    chosen = np.concatenate(
        (
            generator.choice(surface, size=surface_count, replace=False),
            generator.choice(volume, size=count - surface_count, replace=False),
        )
    )
    return np.sort(chosen)


def sampled_graph(points: np.ndarray, neighbors: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= neighbors:
        raise ValueError("sample must contain more nodes than neighbors")
    distance = np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :neighbors]
    source = np.repeat(np.arange(len(points)), neighbors)
    target = nearest.reshape(-1)
    relative = points[target] - points[source]
    return (
        np.vstack((source, target)).astype(np.int64),
        np.column_stack((relative, np.linalg.norm(relative, axis=1))).astype(np.float32),
    )


def fit_normalization(
    dataset_root: Path, case_ids: list[str], nodes_per_case: int, seed: int
) -> Normalization:
    feature_moments: RunningMoments | None = None
    target_moments = RunningMoments(4)
    for position, case_id in enumerate(case_ids):
        case = load_case(case_directory(dataset_root, case_id))
        indices = sample_indices(case, nodes_per_case, seed + position)
        if feature_moments is None:
            feature_moments = RunningMoments(case.features.shape[1])
        feature_moments.update(case.features[indices])
        target_moments.update(case.targets[indices])
    if feature_moments is None:
        raise ValueError("normalization requires at least one training case")
    feature_mean, feature_scale = feature_moments.finish()
    target_mean, target_scale = target_moments.finish()
    return Normalization(feature_mean, feature_scale, target_mean, target_scale)


def build_model(config: ExperimentConfig, input_dim: int):
    if config.model == "pointwise_mlp":
        return pointwise_mlp(input_dim, config.hidden_dim)
    if config.model == "mesh_graph_net":
        return mesh_graph_net(
            input_dim, hidden_dim=config.hidden_dim, layers=config.processor_layers
        )
    return point_neural_operator(input_dim, config.hidden_dim, config.modes)


def _forward(model, model_name: str, x, edge_index=None, edge_features=None):
    return model(x, edge_index, edge_features) if model_name == "mesh_graph_net" else model(x)


def _case_tensors(case, normalization: Normalization, indices: np.ndarray, device, model_name: str):
    import torch

    x = torch.from_numpy(normalization.transform_features(case.features[indices])).to(device)
    y = torch.from_numpy(normalization.transform_targets(case.targets[indices])).to(device)
    if model_name != "mesh_graph_net":
        return x, y, None, None
    edges, edge_features = sampled_graph(case.points[indices])
    return x, y, torch.from_numpy(edges).to(device), torch.from_numpy(edge_features).to(device)


def evaluate_cases(
    model,
    model_name: str,
    dataset_root: Path,
    case_ids: list[str],
    normalization: Normalization,
    nodes_per_case: int,
    seed: int,
    device,
    full_mesh: bool = False,
) -> dict[str, object]:
    import torch

    per_case = []
    started = perf_counter()
    model.eval()
    with torch.inference_mode():
        for position, case_id in enumerate(case_ids):
            directory = case_directory(dataset_root, case_id)
            case = load_case(directory)
            indices = (
                np.arange(len(case.points))
                if full_mesh
                else sample_indices(case, nodes_per_case, seed + 100_000 + position)
            )
            if full_mesh and model_name == "mesh_graph_net":
                graph = load_graph(directory)
                x = torch.from_numpy(normalization.transform_features(case.features)).to(device)
                edges = torch.from_numpy(graph.edge_index).to(device)
                edge_features = torch.from_numpy(graph.edge_features).to(device)
            else:
                x, _, edges, edge_features = _case_tensors(
                    case, normalization, indices, device, model_name
                )
            prediction = _forward(model, model_name, x, edges, edge_features).cpu().numpy()
            physical = normalization.inverse_targets(prediction)
            result = {"case_id": case_id, **field_metrics(case.targets[indices], physical)}
            if full_mesh:
                predicted_force = integrate_airfrans_forces(directory, physical)
                reference_force = integrate_airfrans_forces(directory)
                result["force_coefficients"] = {
                    "prediction": asdict(predicted_force),
                    "reference": asdict(reference_force),
                    "absolute_error": {
                        "drag": abs(predicted_force.drag - reference_force.drag),
                        "lift": abs(predicted_force.lift - reference_force.lift),
                    },
                }
            per_case.append(result)
    elapsed = perf_counter() - started
    fields = ("velocity_x", "velocity_y", "pressure", "turbulent_viscosity")
    aggregate = {
        metric: {
            field: float(np.mean([case[metric][field] for case in per_case])) for field in fields
        }
        for metric in ("rmse", "mae", "relative_l2")
    }
    summary = {
        "case_count": len(per_case),
        "full_mesh": full_mesh,
        "elapsed_seconds": elapsed,
        "mean": aggregate,
        "per_case": per_case,
    }
    if full_mesh:
        summary["mean_force_absolute_error"] = {
            component: float(
                np.mean(
                    [case["force_coefficients"]["absolute_error"][component] for case in per_case]
                )
            )
            for component in ("drag", "lift")
        }
    return summary


def run_experiment(
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    max_train_cases: int | None = None,
    max_validation_cases: int | None = None,
    max_test_cases: int | None = None,
    resume: bool = False,
    strategy: str = "single",
) -> dict[str, object]:
    import torch
    import torch.distributed as dist

    config.validate()
    distributed = context_from_environment(strategy)
    if distributed.enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(distributed.local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if distributed.enabled and torch.cuda.is_available():
        device = torch.device("cuda", distributed.local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_ids, validation_ids, test_ids = official_split(
        manifest_path, config.task, config.validation_cases
    )
    if max_train_cases:
        train_ids = train_ids[:max_train_cases]
    if max_validation_cases:
        validation_ids = validation_ids[:max_validation_cases]
    if max_test_cases:
        test_ids = test_ids[:max_test_cases]
    normalization = fit_normalization(dataset_root, train_ids, config.nodes_per_case, config.seed)
    base_model = build_model(config, len(normalization.feature_mean)).to(device)
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(2, config.patience // 4)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    start_epoch = 0
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        resumed = torch.load(checkpoint_path, map_location=device, weights_only=True)
        base_model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
        best_loss = float(resumed["validation_mean_relative_l2"])
    if distributed.enabled:
        ddp_devices = [distributed.local_rank] if device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=ddp_devices,
            output_device=distributed.local_rank if device.type == "cuda" else None,
        )
    else:
        model = base_model
    started = perf_counter()
    processed_training_nodes = 0
    for epoch in range(start_epoch, config.epochs):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(len(train_ids))
        selected = order[: min(config.cases_per_epoch, len(order))]
        if distributed.enabled:
            selected = partition_equal(selected, distributed.rank, distributed.world_size)
        train_losses = []
        for position in selected:
            case = load_case(case_directory(dataset_root, train_ids[position]))
            indices = sample_indices(
                case, config.nodes_per_case, config.seed + epoch * len(train_ids) + int(position)
            )
            x, y, edges, edge_features = _case_tensors(
                case, normalization, indices, device, config.model
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=config.mixed_precision and device.type == "cuda"
            ):
                loss = torch.nn.functional.mse_loss(
                    _forward(model, config.model, x, edges, edge_features), y
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            processed_training_nodes += len(indices)
        local_training_sum = torch.tensor(
            [sum(train_losses), len(train_losses)], dtype=torch.float64, device=device
        )
        if distributed.enabled:
            dist.all_reduce(local_training_sum, op=dist.ReduceOp.SUM)
            dist.barrier()
        mean_training_loss = float((local_training_sum[0] / local_training_sum[1]).detach().cpu())
        validation_loss_payload = [None]
        if distributed.primary:
            validation = evaluate_cases(
                base_model,
                config.model,
                dataset_root,
                validation_ids,
                normalization,
                config.nodes_per_case,
                config.seed + epoch,
                device,
            )
            validation_loss_payload[0] = float(
                np.mean(list(validation["mean"]["relative_l2"].values()))
            )
        if distributed.enabled:
            dist.broadcast_object_list(validation_loss_payload, src=0)
        validation_loss = float(validation_loss_payload[0])
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - config.minimum_delta
        history.append(
            {
                "epoch": epoch + 1,
                "training_mse": mean_training_loss,
                "validation_mean_relative_l2": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "improved": improved,
            }
        )
        if improved:
            best_loss, stale_epochs = validation_loss, 0
            if distributed.primary:
                torch.save(
                    {
                        "model": base_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "config": asdict(config),
                        "normalization": normalization.to_dict(),
                        "validation_mean_relative_l2": best_loss,
                    },
                    checkpoint_path,
                )
        else:
            stale_epochs += 1
        if distributed.enabled:
            dist.barrier()
        if stale_epochs >= config.patience:
            break
    if distributed.enabled:
        dist.barrier()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    base_model.load_state_dict(checkpoint["model"])
    processed_nodes_tensor = torch.tensor(
        processed_training_nodes, dtype=torch.int64, device=device
    )
    if distributed.enabled:
        dist.all_reduce(processed_nodes_tensor, op=dist.ReduceOp.SUM)
    processed_training_nodes = int(processed_nodes_tensor.cpu())
    if not distributed.primary:
        dist.barrier()
        dist.destroy_process_group()
        return {
            "evidence_label": "distributed_worker_complete",
            "rank": distributed.rank,
            "world_size": distributed.world_size,
        }
    test = evaluate_cases(
        base_model,
        config.model,
        dataset_root,
        test_ids,
        normalization,
        config.nodes_per_case,
        config.seed,
        device,
        full_mesh=not any((max_train_cases, max_validation_cases, max_test_cases)),
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    elapsed = perf_counter() - started
    result = {
        "evidence_label": "airfrans_bounded_run"
        if any((max_train_cases, max_validation_cases, max_test_cases))
        else "airfrans_official_split_result",
        "config": asdict(config),
        "device": str(device),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "parameters": parameter_count(base_model),
        "distributed": {
            "strategy": strategy,
            "world_size": distributed.world_size,
        },
        "splits": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "normalization": normalization.to_dict(),
        "history": history,
        "best_validation_mean_relative_l2": best_loss,
        "test": test,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "elapsed_seconds": elapsed,
        "training_nodes_processed": processed_training_nodes,
        "training_nodes_per_second": processed_training_nodes / elapsed if elapsed else None,
        "peak_accelerator_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "resumed_from_epoch": start_epoch if resume else None,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if distributed.enabled:
        dist.barrier()
        dist.destroy_process_group()
    return result
