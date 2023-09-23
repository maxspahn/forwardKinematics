import casadi as ca
import numpy as np
import pytest

from forwardkinematics import GenericURDFFk
from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError

@pytest.fixture
def fk() -> GenericURDFFk:
    with open("panda.urdf", "r") as file:
        urdf = file.read()
    fk_panda = GenericURDFFk(
        urdf,
        rootLink = 'panda_link0',
        end_link="panda_leftfinger",
    )
    return fk_panda

def test_pandaFk(fk):
    q_ca = ca.SX.sym("q", 7)
    q_np = np.random.random(7)
    fkCasadi = fk.fk(q_ca, 'panda_link9', position_only=False)
    fkNumpy = fk.fk(q_np, 'panda_link9', position_only=False)
    assert isinstance(fkCasadi, ca.SX)
    assert isinstance(fkNumpy, np.ndarray)

def test_pandaFkByName(fk):
    q_ca = ca.SX.sym('q', 7)
    q_np = np.random.random(7)
    fkCasadi = fk.fk_by_name(q_ca, 'panda_link3', position_only=False)
    assert isinstance(fkCasadi, ca.SX)

def test_pandafkByWrongName(fk):
    q_ca = ca.SX.sym('q', 7)
    with pytest.raises(LinkNotInURDFError):
        fkCasadi = fk.fk_by_name(q_ca, 'panda_link10', position_only=False)

def test_simpleFk(fk: GenericURDFFk):
    q_np = np.array([0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000])
    fkNumpy = fk.numpy(q_np, 'panda_link0', position_only=True)
    x_ee = np.array([0.0, 0.0, 0.0])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'panda_link3', position_only=True)
    x_ee = np.array([0.2713, 0.0, 0.4950])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'panda_link9', position_only=True)
    x_ee = np.array([0.4, 0.0, 0.71])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
