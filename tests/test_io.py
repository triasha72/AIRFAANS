from airfaans.io import load_case, save_case
from airfaans.synthetic import make_case


def test_case_round_trip(tmp_path):
    expected = make_case(radial_count=2, angular_count=8)
    path = tmp_path / "case.npz"
    save_case(path, expected)
    actual = load_case(path)
    assert actual.case_id == expected.case_id
    assert actual.features.shape == expected.features.shape
    assert actual.targets.shape == expected.targets.shape
