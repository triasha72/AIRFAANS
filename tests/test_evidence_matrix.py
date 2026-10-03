from argparse import Namespace
from pathlib import Path

from scripts.run_evidence_matrix import command


def test_evidence_matrix_command_is_resumable_and_task_scoped() -> None:
    args = Namespace(
        dataset_root=Path("/data/airfrans"),
        manifest=Path("data/manifests/tasks.json"),
        config=Path("configs/experiment.yaml"),
        output_root=Path("artifacts/matrix"),
        model="mesh_graph_net",
    )
    result = command(args, "aoa_ood", 41)
    assert result[-1] == "--resume"
    assert "artifacts/matrix/aoa_ood/mesh_graph_net-seed41" in result
    assert "aoa_ood" in result
