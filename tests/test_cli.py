from airfaans.cli import training_summary


def test_training_summary_ignores_non_primary_ddp_worker() -> None:
    assert training_summary({"evidence_label": "distributed_worker_complete"}) is None


def test_training_summary_preserves_rank_zero_evidence() -> None:
    result = {
        "evidence_label": "airfrans_bounded_run",
        "device": "cuda:0",
        "best_validation_mean_relative_l2": 0.5,
        "checkpoint": {"path": "best.pt"},
    }
    assert training_summary(result) == result
