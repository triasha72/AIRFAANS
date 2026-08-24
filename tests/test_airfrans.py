import pytest

from airfaans.airfrans import parse_condition


def test_parse_condition_matches_official_name_convention():
    condition = parse_condition("airFoil2D_SST_36.622_11.319_3.941_5.424_1.0_16.283")
    assert condition.inlet_speed == pytest.approx(36.622)
    assert condition.angle_of_attack_deg == pytest.approx(11.319)
    assert 2.0e6 < condition.reynolds < 3.0e6


def test_parse_condition_rejects_unrelated_name():
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_condition("not-an-airfrans-case")
