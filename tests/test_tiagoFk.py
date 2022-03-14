import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.tiagoFk import TiagoFk
import pytest


@pytest.fixture
def fk():
    return TiagoFk(20)


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
    assert fkNumpy[0] == pytest.approx(-0.062, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.193, abs=1e-4)
    fkNumpy = fk.fk(q_np, 4, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(-0.062, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.193 + 0.597, abs=1e-4)


def test_fkNonZeros(fk):
    ee_offset = -0.062
    q_np = np.zeros(fk.n())
    q_np[0:3] = np.array([0.5, 0.3, 0.7])
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 1, positionOnly=True)
    assert fkNumpy[0] == pytest.approx(q_np[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(q_np[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.0, abs=1e-4)
    x = np.array(
        [
            q_np[0] + ee_offset * np.cos(q_np[2]),
            q_np[1] + ee_offset * np.sin(q_np[2]),
            0.0,
        ]
    )
    fkNumpy = fk.fk(q_np, 2, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.193, abs=1e-4)
    fkNumpy = fk.fk(q_np, 4, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.193 + 0.597, abs=1e-4)


def test_arm(fk):
    q_np = np.zeros(fk.n())
    q_np[5] = -1.0
    fkNumpy = fk.fk(q_np, 11, positionOnly=True)
    x = np.array([-0.0544357, 0.676733, 1.15136])
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)


def test_arms_equal(fk):
    """ Tests if both end-effectors have the same z coordinate and opposite y coordinate."""
    q_np = np.zeros(fk.n())
    fkNumpy_left = fk.fk(q_np, 11, positionOnly=True)
    fkNumpy_right = fk.fk(q_np, 18, positionOnly=True)
    assert fkNumpy_right[1] == pytest.approx(-fkNumpy_left[1], abs=1e-4)
    assert fkNumpy_right[2] == pytest.approx(fkNumpy_left[2], abs=1e-4)
