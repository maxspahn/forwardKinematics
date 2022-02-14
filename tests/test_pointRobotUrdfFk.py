import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.pointRobotUrdfFk import PointRobotUrdfFk
import pytest


@pytest.fixture
def fk():
    return PointRobotUrdfFk(3)


def test_fkZeros(fk):
    q_np = np.zeros(fk.n())
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 1, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 2, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)


def test_fkNonZeros(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, positionOnly=True)
    x = np.array(
        [
            q_np[0],
            q_np[1],
            0.0,
        ]
    )
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)
    fkNumpy = fk.fk(q_np, 2, positionOnly=True)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)

def test_fkNonZeros_rotation(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, positionOnly=False)
    x = np.array(
        [
            q_np[0],
            q_np[1],
            0.0,
        ]
    )
    fkNumpy_trans = fkNumpy[0:3, 3]
    fkNumpy_rot = fkNumpy[0:3, 0:3]
    assert fkNumpy_trans[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy_trans[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy_trans[2] == pytest.approx(x[2], abs=1e-4)
    assert fkNumpy_rot[0,0] == pytest.approx(np.cos(q_np[2]), abs=1e-4)
    assert fkNumpy_rot[1,1] == pytest.approx(np.cos(q_np[2]), abs=1e-4)
    assert fkNumpy_rot[0,1] == pytest.approx(-np.sin(q_np[2]), abs=1e-4)
    assert fkNumpy_rot[1,0] == pytest.approx(np.sin(q_np[2]), abs=1e-4)
