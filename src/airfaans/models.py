"""Field-surrogate model families.

Torch is imported lazily so data preparation and evaluation remain usable in a
CPU-only minimal installation.
"""

from __future__ import annotations


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install AIRFAANS with the 'ml' extra to use models.") from exc
    return torch, nn


def pointwise_mlp(input_dim: int, hidden_dim: int = 128, output_dim: int = 4):
    """Pointwise baseline that intentionally ignores graph connectivity."""
    _, nn = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def mesh_graph_net(
    node_dim: int,
    edge_dim: int = 3,
    hidden_dim: int = 128,
    layers: int = 6,
    output_dim: int = 4,
):
    """Build a MeshGraphNet-style residual message-passing model."""
    torch, nn = _torch()

    class ProcessorBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, nodes, edge_index, edge_features):
            source, target = edge_index
            messages = self.edge_mlp(
                torch.cat((nodes[source], nodes[target], edge_features), dim=-1)
            )
            aggregate = torch.zeros_like(nodes)
            aggregate.index_add_(0, target, messages)
            degree = torch.bincount(target, minlength=len(nodes)).clamp_min(1).unsqueeze(-1)
            aggregate = aggregate / degree
            return nodes + self.node_mlp(torch.cat((nodes, aggregate), dim=-1))

    class MeshGraphNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(node_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.processors = nn.ModuleList(ProcessorBlock() for _ in range(layers))
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, nodes, edge_index, edge_features):
            encoded = self.encoder(nodes)
            for processor in self.processors:
                encoded = processor(encoded, edge_index, edge_features)
            return self.decoder(encoded)

    return MeshGraphNet()


def point_neural_operator(
    input_dim: int,
    hidden_dim: int = 128,
    modes: int = 32,
    output_dim: int = 4,
):
    """Build a low-rank global point operator for irregular meshes.

    Learned modes aggregate information across every node, making this a compact
    operator-learning treatment rather than a gridded FNO implementation.
    """
    torch, nn = _torch()

    class PointOperator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lift = nn.Linear(input_dim, hidden_dim)
            self.keys = nn.Linear(hidden_dim, modes)
            self.values = nn.Linear(hidden_dim, hidden_dim)
            self.project = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, nodes):
            latent = torch.nn.functional.silu(self.lift(nodes))
            weights = torch.softmax(self.keys(latent), dim=0)
            global_modes = weights.transpose(0, 1) @ self.values(latent)
            reconstructed = weights @ global_modes
            return self.project(torch.cat((latent, reconstructed), dim=-1))

    return PointOperator()


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
