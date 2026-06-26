import calc
import pytest

def test_calc():
    """
    Tests a standard case of inputs for calc.Visibility()
    """
    v = calc.Visiblity(0.5,1.5)
    assert v.radius_ratio == 0.5
    assert v.satellite_orbit_ratio == 1.5

def test_invalid_radius():
    """
    Tests out an invalid case of radius_ratio in calc.Visibility()
    """
    with pytest.raises(ValueError):
        calc.Visibility(1.5,2.0)

def test_invalid_satellite_ratio():
    """
    Tests out an invalid case of satellite_orbit_ratio in calc.Visiblity()
    """
    with pytest.raises(ValueError):
        calc.Visibility(0.5,0.5)