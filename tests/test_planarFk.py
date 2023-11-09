import casadi as ca
import numpy as np
import pytest
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


def test_planarFk():
    q_ca = ca.SX.sym("q", 2)
    fkPlanar = PlanarArmFk(n=2)
    q_np = np.array([1.0, 0.0])
    fkCasadi = fkPlanar.casadi(q_ca, 2, position_only=False)
    fkNumpy = fkPlanar.casadi(q_np, 2, position_only=False)
    assert fkNumpy[0,2] == np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0
    assert fkNumpy[1,2] == np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0
    assert fkNumpy[0,0] == np.cos(1.0)
    fkCasadi_fun = ca.Function('test_fun', [q_ca], [fkCasadi])
    test_eval = np.array(fkCasadi_fun(q_np))
    assert test_eval[0,2] == pytest.approx(np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0)
    assert test_eval[1,2] == pytest.approx(np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0)
    assert test_eval[0,0] == np.cos(1.0)

def test_planarFk_by_name():
    q_ca = ca.SX.sym("q", 2)
    fkPlanar = PlanarArmFk(n=2)
    q_np = np.array([1.0, 0.0])
    fkCasadi = fkPlanar.casadi(q_ca, "link2", position_only=False)
    fkNumpy = fkPlanar.numpy(q_np, "link2", position_only=False)
    assert fkNumpy[0,2] == np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0
    assert fkNumpy[1,2] == np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0
    assert fkNumpy[0,0] == np.cos(1.0)
    fkCasadi_fun = ca.Function('test_fun', [q_ca], [fkCasadi])
    test_eval = np.array(fkCasadi_fun(q_np))
    assert test_eval[0,2] == pytest.approx(np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0)
    assert test_eval[1,2] == pytest.approx(np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0)
    assert test_eval[0,0] == np.cos(1.0)
