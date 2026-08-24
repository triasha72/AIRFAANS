"""Leakage-aware simulation-level split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from airfaans.data import FlowCase

SplitTask = Literal["interpolation", "scarce", "reynolds_ood", "aoa_ood"]


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = (set(self.train), set(self.validation), set(self.test))
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("simulation IDs cannot appear in more than one split")


def build_split(cases: list[FlowCase], task: SplitTask, seed: int = 17) -> Split:
    if len(cases) < 5:
        raise ValueError("at least five simulations are required")
    ordered = sorted(cases, key=lambda case: case.case_id)
    if task == "reynolds_ood":
        ordered = sorted(ordered, key=lambda case: (case.reynolds, case.case_id))
        test_count = max(1, len(ordered) // 5)
        train_pool, test = ordered[:-test_count], ordered[-test_count:]
    elif task == "aoa_ood":
        ordered = sorted(ordered, key=lambda case: (case.angle_of_attack_deg, case.case_id))
        test_count = max(1, len(ordered) // 5)
        train_pool, test = ordered[:-test_count], ordered[-test_count:]
    else:
        indices = np.random.default_rng(seed).permutation(len(ordered))
        test_count = max(1, len(ordered) // 5)
        test = [ordered[index] for index in indices[:test_count]]
        train_pool = [ordered[index] for index in indices[test_count:]]
    validation_count = max(1, len(train_pool) // 5)
    validation, train = train_pool[-validation_count:], train_pool[:-validation_count]
    if task == "scarce":
        train = train[: max(1, len(train) // 10)]
    split = Split(
        train=tuple(case.case_id for case in train),
        validation=tuple(case.case_id for case in validation),
        test=tuple(case.case_id for case in test),
    )
    split.validate()
    return split
