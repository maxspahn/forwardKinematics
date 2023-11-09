import os
import numpy as np
import pytest

from forwardkinematics import GenericURDFFk


@pytest.fixture
def fk() -> GenericURDFFk:
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/boxer.urdf"
    with open(urdf_file, "r") as file:
        urdf = file.read()
    fk_boxer = GenericURDFFk(
        urdf,
        root_link = 'base_link',
        end_links="ee_link",
        base_type='diffdrive',
    )
    return fk_boxer


def test_fkZeros(fk: GenericURDFFk):
    q_np = np.zeros(fk.n())
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(0, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'ee_link', position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(-0.4, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)
    fkNumpy = fk.numpy(q_np, 'ee_link', position_only=True)
    assert isinstance(fkNumpy, np.ndarray)
    assert fkNumpy[0] == pytest.approx(0.0, abs=1e-4)
    assert fkNumpy[1] == pytest.approx(-0.4, abs=1e-4)
    assert fkNumpy[2] == pytest.approx(0, abs=1e-4)


def test_fkNonZeros(fk: GenericURDFFk):
    ee_offset = 0.4
    q_np = np.array([0.5, 0.3, 0.0])
    fkNumpy = fk.numpy(q_np, 'base_link', position_only=True)
    assert fkNumpy[0] == 0
    assert fkNumpy[1] == 0
    assert fkNumpy[2] == 0
    fkNumpy = fk.numpy(q_np, 'ee_link', position_only=True)
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
    fkNumpy = fk.numpy(q_np, 'ee_link', position_only=True)
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
