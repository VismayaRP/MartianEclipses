import calc
import pytest

def test_calc():
    v = calc.Visiblity(0.5,1.5)
    assert v.radius_ratio == 0.5
    assert v.satellite_orbit_ratio == 1.5

def test_invalid_radius():
    with pytest.raises(ValueError):
        calc.Visibility(1.5,2.0)

def test_invalid_satellite_ratio():
    with pytest.raises(ValueError):
        calc.Visibility(0.5,0.5)