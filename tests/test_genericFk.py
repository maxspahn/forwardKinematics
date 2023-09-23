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
    with open(absolute_path + "/albert_prismatic_fingers.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link0', end_link='panda_vacuum')
    assert fk.n() == 7


@pytest.fixture
def generic_fk_clean():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link0', end_link='panda_ee')
    return fk

@pytest.fixture
def generic_fk_fixed_fingers():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert_fixed_fingers.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link0', end_link='panda_vacuum')
    return fk

@pytest.fixture
def generic_fk_prismatic_fingers():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert_prismatic_fingers.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link0', end_link='panda_vacuum')
    return fk

def test_zeros(generic_fk_clean):
    generic_fk = generic_fk_clean
    n = generic_fk.n()
    assert n == 7
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = generic_fk.fk(
        q_ca, "panda_link0", "panda_link0", position_only=True
    )
    fk_panda_link_np = generic_fk.fk(
        q_np, "panda_link0", "panda_link0", position_only=True
    )
    assert isinstance(fk_panda_link, ca.SX)
    assert isinstance(fk_panda_link_np, np.ndarray)
    print(fk_panda_link)
    assert fk_panda_link[0] == ca.SX(0.0)
    assert fk_panda_link[1] == ca.SX(0.0)
    assert fk_panda_link[2] == ca.SX(0.0)
    assert fk_panda_link_np[0] == pytest.approx(0.0)
    assert fk_panda_link_np[1] == pytest.approx(0.0)
    assert fk_panda_link_np[2] == pytest.approx(0.0)

def test_clean(generic_fk_clean):
    generic_fk = generic_fk_clean
    n = generic_fk.n()
    assert n == 7
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = generic_fk.fk(
        q_ca, "panda_link7", "panda_ee", position_only=True
    )
    fk_panda_link_np = generic_fk.fk(
        q_np, "panda_link0", "panda_ee", position_only=True
    )
    assert isinstance(fk_panda_link, ca.SX)
    assert isinstance(fk_panda_link_np, np.ndarray)
 
def test_fixed_fingers(generic_fk_fixed_fingers):
    generic_fk = generic_fk_fixed_fingers
    n = generic_fk.n()
    assert n == 7
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = generic_fk.fk(
        q_ca, "panda_link7", "panda_vacuum", position_only=True
    )
    fk_panda_link_np = generic_fk.fk(
        q_np, "panda_link0", "panda_vacuum", position_only=True
    )
    assert isinstance(fk_panda_link, ca.SX)
    assert isinstance(fk_panda_link_np, np.ndarray)


def test_prismatic_fingers(generic_fk_prismatic_fingers):
    generic_fk = generic_fk_prismatic_fingers
    n = generic_fk.n()
    assert n == 7
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = generic_fk.fk(
        q_ca, "panda_link7", "panda_vacuum", position_only=True
    )
    fk_panda_link_np = generic_fk.fk(
        q_np, "panda_link0", "panda_vacuum", position_only=True
    )
    assert isinstance(fk_panda_link, ca.SX)
    assert isinstance(fk_panda_link_np, np.ndarray)

def test_compare(generic_fk_prismatic_fingers, panda_fk):
    generic_fk = generic_fk_prismatic_fingers
    assert generic_fk.n() == panda_fk.n()
    # casadi comparison
    q_ca = ca.SX.sym('q', panda_fk.n())
    fk_panda = panda_fk.fk(q_ca, 1, position_only=True)
    fk_generic = generic_fk.fk(q_ca, "panda_link0", "panda_link3", position_only=True)
    q_panda = ca.symvar(fk_panda)
    q_generic = ca.symvar(fk_generic)
    assert ca.is_equal(q_panda[0], q_generic[0])

    # numpy comparison
    q_np = np.random.random(panda_fk.n())
    fk_panda = panda_fk.fk(q_np, 1, position_only=True)
    fk_generic = generic_fk.fk(q_np, "panda_link0", "panda_link3", position_only=True)
    assert fk_panda[0] == pytest.approx(fk_generic[0])

    


