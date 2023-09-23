import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.pointRobotUrdfFk import PointRobotUrdfFk
from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError
import pytest


@pytest.fixture
def fk():
    return PointRobotUrdfFk()


def test_fkZeros(fk):
    q_np = np.zeros(fk.n())
    fkNumpy = fk.fk(q_np, 0, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 1, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 2, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)


def test_fkNonZeros(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    fkNumpy = fk.fk(q_np, 0, position_only=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, position_only=True)
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
    fkNumpy = fk.fk(q_np, 2, position_only=True)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)

def test_fkNonZeros_rotation(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    fkNumpy = fk.fk(q_np, 0, position_only=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, position_only=False)
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

def test_error_raise(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    with pytest.raises(LinkNotInURDFError):
        fkNumpy = fk.fk(q_np, "panda_link3", position_only=True)
