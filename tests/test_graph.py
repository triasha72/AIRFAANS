import numpy as np

from airfaans.graph import from_cells, knn_graph
from airfaans.synthetic import make_case


def test_knn_graph_has_expected_shape_and_no_self_edges():
    case = make_case(radial_count=4, angular_count=8)
    graph = knn_graph(case, neighbors=3)
    assert graph.edge_index.shape == (2, 32 * 3)
    assert graph.edge_features.shape == (32 * 3, 3)
    assert not np.any(graph.edge_index[0] == graph.edge_index[1])
    assert np.all(graph.edge_features[:, 2] > 0)


def test_cell_graph_is_bidirectional():
    case = make_case(radial_count=2, angular_count=4)
    graph = from_cells(case, np.array([[0, 1, 2], [0, 2, 3]]))
    edges = set(map(tuple, graph.edge_index.T.tolist()))
    assert all((target, source) in edges for source, target in edges)
