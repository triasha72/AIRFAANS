from airfaans.active_learning import select_high_uncertainty, select_random


def test_uncertainty_selection_is_ranked_and_tie_broken_by_id():
    selected = select_high_uncertainty(["c", "a", "b"], [0.2, 0.9, 0.9], 2)
    assert selected == ["a", "b"]


def test_random_selection_is_seeded():
    case_ids = [f"case-{index}" for index in range(10)]
    assert select_random(case_ids, 3, 17) == select_random(case_ids, 3, 17)
