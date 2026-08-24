from airfaans.splits import build_split
from airfaans.synthetic import make_case


def cases():
    return [
        make_case(
            case_id=f"case-{index:02d}",
            reynolds=5e5 + index * 1e5,
            angle_of_attack_deg=-2 + index,
            radial_count=2,
            angular_count=8,
        )
        for index in range(10)
    ]


def test_reynolds_ood_holds_out_highest_reynolds_numbers():
    materialized = cases()
    split = build_split(materialized, "reynolds_ood")
    by_id = {case.case_id: case for case in materialized}
    assert min(by_id[case_id].reynolds for case_id in split.test) > max(
        by_id[case_id].reynolds for case_id in split.train + split.validation
    )


def test_interpolation_is_deterministic_and_disjoint():
    first = build_split(cases(), "interpolation", seed=3)
    second = build_split(cases(), "interpolation", seed=3)
    assert first == second
    assert not (set(first.train) & set(first.test))
