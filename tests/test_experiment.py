import json
from dataclasses import replace

import numpy as np

from airfaans.distributed import partition_equal
from airfaans.experiment import (
    ExperimentConfig,
    config_from_yaml,
    official_split,
    sample_indices,
    sampled_graph,
)
from airfaans.synthetic import make_case


def test_official_split_reserves_validation_without_touching_test(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "full_train": [f"train-{index}" for index in range(8)],
                    "full_test": ["test-1", "test-2"],
                }
            }
        )
    )
    train, validation, test = official_split(manifest, "interpolation", 2)
    assert train == [f"train-{index}" for index in range(6)]
    assert validation == ["train-6", "train-7"]
    assert test == ["test-1", "test-2"]
    assert not (set(train) & set(validation) | set(train) & set(test) | set(validation) & set(test))


def test_sampling_is_deterministic_and_preserves_surface_nodes():
    case = make_case(radial_count=4, angular_count=32)
    first = sample_indices(case, 64, seed=17)
    second = sample_indices(case, 64, seed=17)
    np.testing.assert_array_equal(first, second)
    assert case.surface_mask[first].any()
    edges, attributes = sampled_graph(case.points[first], neighbors=4)
    assert edges.shape == (2, 256)
    assert attributes.shape == (256, 3)


def test_experiment_config_validation():
    config = ExperimentConfig(model="pointwise_mlp")
    config.validate()
    try:
        replace(config, model="unknown").validate()
    except ValueError as error:
        assert "model must be" in str(error)
    else:
        raise AssertionError("invalid model was accepted")


def test_yaml_configuration_resolves_one_treatment(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
models:
  mesh_graph_net: {hidden_dim: 64, processor_layers: 3}
training:
  epochs: 5
  learning_rate: 0.002
  mixed_precision: false
  nodes_per_case: 128
  cases_per_epoch: 2
  validation_cases: 1
  early_stopping_patience: 3
"""
    )
    config = config_from_yaml(path, "mesh_graph_net", "interpolation", 29)
    assert (config.hidden_dim, config.processor_layers, config.seed) == (64, 3, 29)


def test_distributed_partition_has_equal_disjoint_work():
    indices = np.arange(7)
    rank_zero = partition_equal(indices, rank=0, world_size=2)
    rank_one = partition_equal(indices, rank=1, world_size=2)
    np.testing.assert_array_equal(rank_zero, [0, 2, 4])
    np.testing.assert_array_equal(rank_one, [1, 3, 5])
    assert not set(rank_zero) & set(rank_one)
