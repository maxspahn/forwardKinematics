import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.boxerFk import BoxerFk
import pytest

@pytest.fixture
def fk():
    return BoxerFk(3)


def test_fkZeros(fk):
    q_np = np.zeros(fk.n())
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == -0.4
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 2, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == -0.4
    assert fkNumpy[2] == 0

def test_fkNonZeros(fk):
    q_np = np.array([0.5, 0.3, 0.2])
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, positionOnly=True)
    assert fkNumpy[0] == 0.5 + 0.4 * np.sin(0.2)
    assert fkNumpy[1] == 0.3 - 0.4 * np.cos(0.2)
    assert fkNumpy[2] == 0.0
    fkNumpy = fk.fk(q_np, 2, positionOnly=True)
    assert fkNumpy[0] == 0.5 + 0.4 * np.sin(0.2)
    assert fkNumpy[1] == 0.3 - 0.4 * np.cos(0.2)
    assert fkNumpy[2] == 0.0
