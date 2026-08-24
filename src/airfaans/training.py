"""Single-case training primitives with AMP and checkpoint support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    seed: int = 17


def fit(model, forward, target, config: TrainConfig, checkpoint: Path | None = None):
    """Train a model through a caller-supplied forward closure."""
    import torch

    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_epoch = 0
    if checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=target.device, weights_only=True)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
    scaler = torch.amp.GradScaler(
        target.device.type,
        enabled=config.mixed_precision and target.device.type == "cuda",
    )
    losses: list[float] = []
    started = perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.epochs):
        with torch.autocast(
            device_type=target.device.type,
            enabled=config.mixed_precision and target.device.type == "cuda",
        ):
            prediction = forward(model)
            loss = torch.nn.functional.mse_loss(prediction, target)
            scaled_loss = loss / config.gradient_accumulation
        scaler.scale(scaled_loss).backward()
        if (epoch + 1) % config.gradient_accumulation == 0 or epoch + 1 == config.epochs:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                },
                checkpoint,
            )
    return {
        "epochs": config.epochs - start_epoch,
        "final_loss": losses[-1],
        "elapsed_seconds": perf_counter() - started,
        "losses": losses,
    }
