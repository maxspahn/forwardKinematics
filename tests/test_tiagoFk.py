import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.tiagoFk import TiagoFk
import pytest


@pytest.fixture
def fk():
    return TiagoFk()


def test_fkZeros(fk):
    q_np = np.zeros(fk.n())
    fkNumpy = fk.fk(q_np, 0, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 1, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(-0.18, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.1, abs=1e-4)
    fkNumpy = fk.fk(q_np, 2, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(-0.18 - 0.062, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.293, abs=1e-4)
    fkNumpy = fk.fk(q_np, 4, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(-0.18 - 0.062, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.293 + 0.597, abs=1e-4)


def test_fkNonZeros(fk):
    ee_offset = -0.062 - 0.18
    q_np = np.zeros(fk.n())
    q_np[0:3] = np.array([0.5, 0.3, 0.7])
    fkNumpy = fk.fk(q_np, 0, position_only=True)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 1, position_only=True)
    assert fkNumpy[0] == pytest.approx(q_np[0] - 0.18 * np.cos(q_np[2]), abs=1e-4)
    assert fkNumpy[1] == pytest.approx(q_np[1] - 0.18 * np.sin(q_np[2]), abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.1, abs=1e-4)
    x = np.array(
        [
            q_np[0] + ee_offset * np.cos(q_np[2]),
            q_np[1] + ee_offset * np.sin(q_np[2]),
            0.1,
        ]
    )
    fkNumpy = fk.fk(q_np, 2, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.293, abs=1e-4)
    fkNumpy = fk.fk(q_np, 4, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.293 + 0.597, abs=1e-4)


def test_arm(fk):
    q_np = np.zeros(fk.n())
    q_np[7] = -1.0
    fkNumpy = fk.fk(q_np, 11, position_only=True)
    x = np.array([-0.234444, 0.80131, 1.4454])
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)


def test_arms_equal(fk):
    """ Tests if both end-effectors have the same z coordinate and opposite y coordinate."""
    q_np = np.zeros(fk.n())
    fkNumpy_left = fk.fk(q_np, 11, position_only=True)
    fkNumpy_right = fk.fk(q_np, 18, position_only=True)
    assert fkNumpy_right[1] == pytest.approx(-fkNumpy_left[1], abs=1e-4)
    assert fkNumpy_right[2] == pytest.approx(fkNumpy_left[2], abs=1e-4)
