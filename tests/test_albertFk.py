import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.albertFk import AlbertFk
from forwardkinematics.urdfFks.pandaFk import PandaFk
import pytest

@pytest.fixture
def fk():
    return AlbertFk(10)
    

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
    assert fkNumpy[0] == pytest.approx(0.15, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.2895, abs=1e-4)


def test_fkNonZeros(fk):
    q_np = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fkNumpy = fk.fk(q_np, 0, positionOnly=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.fk(q_np, 1, positionOnly=True)
    assert fkNumpy[0] == 0.5 + 0.4 * np.sin(0.2)
    assert fkNumpy[1] == 0.3 - 0.4 * np.cos(0.2)
    assert fkNumpy[2] == 0.0
    fkNumpy = fk.fk(q_np, 2, positionOnly=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[2] == pytest.approx(0.2895, abs=1e-4)

def test_arm(fk):
    q_np = np.array([0, 0, 0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000])
    fkBase = fk.fk(q_np, 3, positionOnly=True)
    fkNumpy = fk.fk(q_np, 3, positionOnly=True)
    fkPanda = fkNumpy - fkBase
    x_ee = np.array([0.0, 0.0, 0.0])
    assert fkPanda[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkPanda[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkPanda[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.fk(q_np, 4, positionOnly=True)
    fkPanda = fkNumpy - fkBase
    x_ee = np.array([0.2713, 0.0, 0.4950])
    assert fkPanda[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkPanda[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkPanda[2] == pytest.approx(x_ee[2], abs=1e-4)

def test_comparePanda(fk):
    fkPanda = PandaFk(7)
    q_np = np.array([0, 0, 0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000])
    fk_base = fk.fk(q_np, 3, positionOnly=True)
    for i in range(7):
        print(i)
        fk_panda = fkPanda.fk(q_np[3:], i, positionOnly=True)
        fk_albert = fk.fk(q_np, i+3, positionOnly=True)
        fk_numpy = fk_albert - fk_base
        for j in range(3):
            assert fk_panda[j] == pytest.approx(fk_numpy[j], abs=1e-4)
