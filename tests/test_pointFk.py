import casadi as ca
import numpy as np
from src.fk.planarFks.poinFk import PointFk


def test_pointFk():
    q_ca = ca.SX.sym("q", 2)
    fkPoint = PointFk(2)
    q_np = np.array([3.0, 0.0])
    fkCasadi = fkPoint.fk(q_ca, 1, positionOnly=True)
    fkNumpy = fkPoint.fk(q_np, 1, positionOnly=True)
    assert fkNumpy[0] == 3.0
    assert fkNumpy[1] == 0.0
    assert fkCasadi[0] == q_ca[0]
    assert fkCasadi[1] == q_ca[1]
