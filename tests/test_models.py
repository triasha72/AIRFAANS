import pytest

from airfaans.models import mesh_graph_net, parameter_count, point_neural_operator, pointwise_mlp

torch = pytest.importorskip("torch")


def test_all_model_families_predict_four_fields():
    node_features = torch.randn(12, 9)
    edge_index = torch.tensor(
        [[index for index in range(12)], [(index + 1) % 12 for index in range(12)]],
        dtype=torch.long,
    )
    edge_features = torch.randn(12, 3)
    models_and_outputs = (
        (pointwise_mlp(9, hidden_dim=16), lambda model: model(node_features)),
        (
            mesh_graph_net(9, hidden_dim=16, layers=2),
            lambda model: model(node_features, edge_index, edge_features),
        ),
        (point_neural_operator(9, hidden_dim=16, modes=4), lambda model: model(node_features)),
    )
    for model, forward in models_and_outputs:
        assert forward(model).shape == (12, 4)
        assert parameter_count(model) > 0
