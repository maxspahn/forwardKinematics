import warnings
import pytest
import numpy as np
import casadi as ca

def blueprint_test(test_main):
    """
    Blueprint for fk tests. The function verifies if the main returns a numpy
    and a casadi instance.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fk_casadi, fk_numpy = test_main()
    assert isinstance(fk_casadi, ca.SX)
    assert isinstance(fk_numpy, np.ndarray)

def test_mounting_tf():
    from generic_using_mount import main
    blueprint_test(main)

def test_robotic_arm():
    from robotic_arm import main
    blueprint_test(main)

def test_holonomic_base():
    from holonomic_base import main
    blueprint_test(main)

def test_mobile_manipulator():
    from mobile_manipulator import main
    blueprint_test(main)

def test_nonholonomic_base():
    from nonholonomic_base import main
    blueprint_test(main)

def test_planar_arm():
    from planar_arm import main
    blueprint_test(main)

