"""Mesh and point-cloud graph construction."""

from __future__ import annotations

import numpy as np

from airfaans.data import FlowCase, GraphCase


def knn_graph(case: FlowCase, neighbors: int = 8) -> GraphCase:
    """Build a deterministic directed k-nearest-neighbor graph."""
    case.validate()
    node_count = len(case.points)
    if not 1 <= neighbors < node_count:
        raise ValueError("neighbors must be between 1 and node_count - 1")
    delta = case.points[:, None, :] - case.points[None, :, :]
    distance_sq = np.sum(delta**2, axis=-1)
    np.fill_diagonal(distance_sq, np.inf)
    nearest = np.argsort(distance_sq, axis=1, kind="stable")[:, :neighbors]
    source = np.repeat(np.arange(node_count), neighbors)
    target = nearest.reshape(-1)
    relative = case.points[target] - case.points[source]
    distances = np.linalg.norm(relative, axis=1, keepdims=True)
    graph = GraphCase(
        flow=case,
        edge_index=np.vstack((source, target)).astype(np.int64),
        edge_features=np.column_stack((relative, distances)).astype(np.float32),
    )
    graph.validate()
    return graph


def from_cells(case: FlowCase, cells: np.ndarray) -> GraphCase:
    """Convert polygon/triangle connectivity to a deduplicated bidirectional graph."""
    edges: set[tuple[int, int]] = set()
    for cell in np.asarray(cells, dtype=np.int64):
        valid = cell[cell >= 0]
        for index, source in enumerate(valid):
            target = valid[(index + 1) % len(valid)]
            if source != target:
                edges.add((int(source), int(target)))
                edges.add((int(target), int(source)))
    ordered = np.asarray(sorted(edges), dtype=np.int64)
    if not len(ordered):
        raise ValueError("cells produced no edges")
    edge_index = ordered.T
    relative = case.points[edge_index[1]] - case.points[edge_index[0]]
    edge_features = np.column_stack((relative, np.linalg.norm(relative, axis=1)))
    graph = GraphCase(case, edge_index, edge_features.astype(np.float32))
    graph.validate()
    return graph


def to_pyg(graph: GraphCase):
    """Convert the internal representation to `torch_geometric.data.Data`."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'ml' extra for PyG conversion.") from exc
    graph.validate()
    return Data(
        x=torch.from_numpy(graph.flow.features),
        y=torch.from_numpy(graph.flow.targets),
        pos=torch.from_numpy(graph.flow.points),
        edge_index=torch.from_numpy(graph.edge_index),
        edge_attr=torch.from_numpy(graph.edge_features),
        surface_mask=torch.from_numpy(graph.flow.surface_mask),
        case_id=graph.flow.case_id,
    )
