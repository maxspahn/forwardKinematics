import casadi as ca
import numpy as np
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


def test_planarFk():
    q_ca = ca.SX.sym("q", 2)
    fkPlanar = PlanarArmFk(2)
    q_np = np.array([1.0, 0.0])
    fkCasadi = fkPlanar.fk(q_ca, 2, positionOnly=False)
    fkNumpy = fkPlanar.fk(q_np, 2, positionOnly=False)
    assert fkNumpy[0] == np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0
    assert fkNumpy[1] == np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0
    assert fkNumpy[2] == 1.0
    fkCasadi_fun = ca.Function('test_fun', [q_ca], [fkCasadi])
    test_eval = fkCasadi_fun(q_np)
    assert test_eval[0] == np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0
    assert test_eval[1] == np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0
    assert test_eval[2] == 1.0
