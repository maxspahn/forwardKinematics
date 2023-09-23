import os
import casadi as ca
import numpy as np
import pytest 

from forwardkinematics import GenericURDFFk


@pytest.fixture
def fk():
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/mobile_panda.urdf"
    with open(urdf_file, "r") as file:
        urdf = file.read()
    fk_panda = GenericURDFFk(
        urdf,
        rootLink = 'world',
        end_link="panda_link9",
    )
    return fk_panda

def test_mobilePandaFk(fk):
    q_ca = ca.SX.sym("q", fk.n())
    q_np = np.random.random(fk.n())
    fkCasadi = fk.casadi(q_ca, 'panda_link9', position_only=False)
    fkNumpy = fk.numpy(q_np, 'panda_link9', position_only=False)
    assert isinstance(fkCasadi, ca.SX)
    assert isinstance(fkNumpy, np.ndarray)

def test_simpleFk(fk: GenericURDFFk):
    q_np = np.array([0.0, 0.0, 0.0, 0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000])
    base_height = 0.25
    fkNumpy = fk.numpy(q_np, 'panda_link0', position_only=True)
    x_ee = np.array([0.0, 0.0, 0.25])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'panda_link1', position_only=True)
    x_ee = np.array([0.2713, 0.0, 0.25 + 0.4950])
    #assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    #assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'panda_link9', position_only=True)
    x_ee = np.array([0.4, 0.0, 0.69 + 0.25])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
