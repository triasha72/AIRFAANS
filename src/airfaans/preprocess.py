"""Parallel VTK preprocessing with an optional Dask execution path."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from airfaans.io import load_vtk_case, save_case


def preprocess_one(spec: tuple[Path, str, float, float, Path]) -> str:
    source, case_id, reynolds, angle, target = spec
    save_case(target, load_vtk_case(source, case_id, reynolds, angle))
    return case_id


def preprocess_many(specs: Iterable[tuple[Path, str, float, float, Path]], parallel: bool = True):
    materialized = list(specs)
    if not parallel:
        return [preprocess_one(spec) for spec in materialized]
    try:
        import dask
    except ImportError as exc:
        raise RuntimeError("Install the 'tracking' extra to use Dask preprocessing.") from exc
    delayed = [dask.delayed(preprocess_one)(spec) for spec in materialized]
    return list(dask.compute(*delayed))
