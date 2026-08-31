from airfaans.active_learning import build_acquisition_round, select_high_uncertainty, select_random


def test_uncertainty_selection_is_ranked_and_tie_broken_by_id():
    selected = select_high_uncertainty(["c", "a", "b"], [0.2, 0.9, 0.9], 2)
    assert selected == ["a", "b"]


def test_random_selection_is_seeded():
    case_ids = [f"case-{index}" for index in range(10)]
    assert select_random(case_ids, 3, 17) == select_random(case_ids, 3, 17)


def test_acquisition_round_records_matched_comparison():
    report = {
        "checkpoint_sha256": ["a", "b", "c"],
        "per_case": [
            {"case_id": "low", "mean_uncertainty": 0.1},
            {"case_id": "high", "mean_uncertainty": 0.9},
        ],
    }
    result = build_acquisition_round(report, 1, 17)
    assert result["uncertainty_selected"] == ["high"]
    assert result["count"] == len(result["random_selected"])
