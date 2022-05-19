import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
import os
import pytest

from forwardkinematics.urdfFks.pandaFk import PandaFk

@pytest.fixture
def panda_fk():
    return PandaFk()

def test_creation():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link0', end_link='panda_ee')
    assert fk.n() == 7


@pytest.fixture
def generic_fk():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link0', end_link='panda_ee')
    return fk


def test_simple(generic_fk):
    n = generic_fk.n()
    assert n == 7
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = generic_fk.fk(
        q_ca, "panda_link7", "panda_ee", positionOnly=True
    )
    fk_panda_link_np = generic_fk.fk(
        q_np, "panda_link0", "panda_ee", positionOnly=True
    )
    assert isinstance(fk_panda_link, ca.SX)
    assert isinstance(fk_panda_link_np, np.ndarray)

def test_compare(generic_fk, panda_fk):
    assert generic_fk.n() == panda_fk.n()
    # casadi comparison
    q_ca = ca.SX.sym('q', panda_fk.n())
    fk_panda = panda_fk.fk(q_ca, 1, positionOnly=True)
    fk_generic = generic_fk.fk(q_ca, "panda_link0", "panda_link3", positionOnly=True)
    q_panda = ca.symvar(fk_panda)
    q_generic = ca.symvar(fk_generic)
    assert ca.is_equal(q_panda[0], q_generic[0])

    # numpy comparison
    q_np = np.random.random(panda_fk.n())
    fk_panda = panda_fk.fk(q_np, 1, positionOnly=True)
    fk_generic = generic_fk.fk(q_np, "panda_link0", "panda_link3", positionOnly=True)
    assert fk_panda[0] == pytest.approx(fk_generic[0])

    


