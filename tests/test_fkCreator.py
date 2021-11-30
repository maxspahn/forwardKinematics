import casadi as ca
import numpy as np
from forwardKinematics.fksCommon.fk_creator import FkCreator

def test_fkCreator():
    fkCreator = FkCreator('planarArm', 3)
    fkPlanar = fkCreator.fk()
    q_np = np.array([1.0, 0.0, 0.0])
    fkNumpy = fkPlanar.fk(q_np, 3, positionOnly=False)
    assert fkNumpy[0] == np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0 + np.cos(1.0) * 1.0
    assert fkNumpy[1] == np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0 + np.sin(1.0) * 1.0
    assert fkNumpy[2] == 1.0
