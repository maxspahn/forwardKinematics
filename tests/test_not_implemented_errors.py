import pytest
import casadi as ca
import numpy as np

from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.planarFks.planar_fk import ForwardKinematicsPlanar

class myFK(ForwardKinematics):
    pass

class myPlanarFK(ForwardKinematicsPlanar):
    pass


def test_myFK():
    fk = myFK()
    with pytest.raises(NotImplementedError):
        fk.casadi(ca.SX.sym('q', 6), 3)
    with pytest.raises(NotImplementedError):
        fk.numpy(np.ones(5), 5)

def test_myPlanarFk():
    fk = myPlanarFK()
    with pytest.raises(NotImplementedError):
        fk.casadi(ca.SX.sym('q', 6), 3)
    with pytest.raises(NotImplementedError):
        fk.numpy(np.ones(5), 5)

