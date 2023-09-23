import os
import numpy as np
import pytest

from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError
from forwardkinematics import GenericURDFFk


@pytest.fixture
def fk():
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/pointRobot.urdf"
    with open(urdf_file, "r") as file:
        urdf = file.read()
    fk_point_robot = GenericURDFFk(
        urdf,
        rootLink = 'origin',
        end_link="base_link",
    )
    return fk_point_robot


def test_fkZeros(fk):
    q_np = np.zeros(fk.n())
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)


def test_fkNonZeros(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    fkNumpy = fk.fk(q_np, 'origin', position_only=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
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
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
    assert fkNumpy[0] == pytest.approx(x[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x[2], abs=1e-4)

def test_fkNonZeros_rotation(fk):
    q_np = np.array([0.5, 0.3, 0.3])
    fkNumpy = fk.numpy(q_np, 'origin', position_only=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=False)
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
