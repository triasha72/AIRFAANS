from argparse import Namespace
from pathlib import Path

from scripts.run_evidence_matrix import command


def test_evidence_matrix_starts_fresh_then_resumes_existing_checkpoint(tmp_path: Path) -> None:
    args = Namespace(
        dataset_root=Path("/data/airfrans"),
        manifest=Path("data/manifests/tasks.json"),
        config=Path("configs/experiment.yaml"),
        output_root=tmp_path,
        model="mesh_graph_net",
    )
    result = command(args, "aoa_ood", 41)
    assert "--resume" not in result
    output = tmp_path / "aoa_ood" / "mesh_graph_net-seed41"
    assert str(output) in result
    assert "aoa_ood" in result
    output.mkdir(parents=True)
    for checkpoint in ("latest.pt", "best.pt"):
        path = output / checkpoint
        path.touch()
        assert command(args, "aoa_ood", 41)[-1] == "--resume"
        path.unlink()
