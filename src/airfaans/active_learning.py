"""Simulation-case selection for active-learning experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_high_uncertainty(
    case_ids: Sequence[str], uncertainty: Sequence[float], count: int
) -> list[str]:
    if len(case_ids) != len(uncertainty):
        raise ValueError("case IDs and uncertainty scores must align")
    if not 0 < count <= len(case_ids):
        raise ValueError("count must select at least one available case")
    ranked = sorted(
        zip(case_ids, np.asarray(uncertainty, dtype=float), strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return [case_id for case_id, _ in ranked[:count]]


def select_random(case_ids: Sequence[str], count: int, seed: int) -> list[str]:
    if not 0 < count <= len(case_ids):
        raise ValueError("count must select at least one available case")
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(case_ids), size=count, replace=False)
    return [case_ids[index] for index in sorted(indices)]
