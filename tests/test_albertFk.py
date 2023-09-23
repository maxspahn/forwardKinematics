import numpy as np
from forwardkinematics.urdfFks.albertFk import AlbertFk
from forwardkinematics.urdfFks.pandaFk import PandaFk
import pytest


@pytest.fixture
def fk():
    return AlbertFk()


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
    assert fkNumpy[1] == pytest.approx(-0.4, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 2, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0.5995, abs=1e-4)


def test_fkNonZeros(fk):
    ee_offset = 0.4
    q_np = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fkNumpy = fk.fk(q_np, 0, position_only=True)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.fk(q_np, 1, position_only=True)
    x = np.array(
        [
            q_np[0] + ee_offset * np.sin(q_np[2]),
            q_np[1] - ee_offset * np.cos(q_np[2]),
            0.0,
        ]
    )
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)
    fkNumpy = fk.fk(q_np, 2, position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[2] == pytest.approx(0.5995, abs=1e-4)


def test_arm(fk):
    q_np = np.array(
        [0, 0, 0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000]
    )
    T_base = fk.fk(q_np, 3, position_only=False)
    T_base_inv = np.linalg.inv(T_base)
    T = fk.fk(q_np, 3, position_only=False)
    fkPanda = np.dot(T_base_inv, T)[0:3, 3]
    x_ee = np.array([0.0, 0.0, 0.0])
    assert fkPanda[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkPanda[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkPanda[2] == pytest.approx(x_ee[2], abs=1e-4)
    T = fk.fk(q_np, 4, position_only=False)
    fkPanda = np.dot(T_base_inv, T)[0:3, 3]
    x_ee = np.array([0.2713, 0.0, 0.4950])
    assert fkPanda[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkPanda[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkPanda[2] == pytest.approx(x_ee[2], abs=1e-4)

def test_arm_by_name(fk):
    q_np = np.array(
        [0, 0, 0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000]
    )
    T_base = fk.fk(q_np, "panda_link0", position_only=False)
    T_base_inv = np.linalg.inv(T_base)
    T = fk.fk(q_np, "panda_link0", position_only=False)
    fkPanda = np.dot(T_base_inv, T)[0:3, 3]
    x_ee = np.array([0.0, 0.0, 0.0])
    assert fkPanda[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkPanda[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkPanda[2] == pytest.approx(x_ee[2], abs=1e-4)
    T = fk.fk(q_np, "panda_link3", position_only=False)
    fkPanda = np.dot(T_base_inv, T)[0:3, 3]
    x_ee = np.array([0.2713, 0.0, 0.4950])
    assert fkPanda[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkPanda[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkPanda[2] == pytest.approx(x_ee[2], abs=1e-4)


def test_comparePanda(fk):
    fkPanda = PandaFk()
    q_np = np.array(
        [0, 0, 0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000]
    )
    T_base = fk.fk(q_np, 3, position_only=False)
    T_base_inv = np.linalg.inv(T_base)
    for i in range(7):
        fk_panda = fkPanda.fk(q_np[3:], i, position_only=True)
        T_albert = fk.fk(q_np, i+3, position_only=False)
        fk_numpy = np.dot(T_base_inv, T_albert)[0:3, 3]
        for j in range(3):
            assert fk_panda[j] == pytest.approx(fk_numpy[j], abs=1e-4)

def test_comparePanda_by_name(fk):
    fkPanda = PandaFk()
    q_np = np.array(
        [0, 0, 0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000]
    )
    T_base = fk.fk(q_np, "panda_link0", position_only=False)
    T_base_inv = np.linalg.inv(T_base)
    for i in range(7):
        fk_panda = fkPanda.fk(q_np[3:], f"panda_link{i}", position_only=True)
        T_albert = fk.fk(q_np, f"panda_link{i}", position_only=False)
        fk_numpy = np.dot(T_base_inv, T_albert)[0:3, 3]
        for j in range(3):
            assert fk_panda[j] == pytest.approx(fk_numpy[j], abs=1e-4)
