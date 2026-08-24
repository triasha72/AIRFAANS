"""Small, explicit primitives for torchrun-based multi-GPU execution."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def primary(self) -> bool:
        return self.rank == 0


def context_from_environment(strategy: str) -> DistributedContext:
    if strategy not in {"single", "ddp"}:
        raise ValueError("strategy must be 'single' or 'ddp'")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if strategy == "single":
        if world_size != 1:
            raise ValueError("torchrun detected but strategy is 'single'")
        return DistributedContext(False, 0, 0, 1)
    if world_size < 2:
        raise ValueError("DDP requires torchrun with at least two processes")
    return DistributedContext(
        True,
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        world_size,
    )


def partition_equal(indices: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Partition an equal number of optimizer steps across ranks."""
    indices = np.asarray(indices)
    usable = len(indices) - len(indices) % world_size
    if usable < world_size:
        raise ValueError("selected cases must include at least one case per rank")
    return indices[:usable][rank:usable:world_size]
