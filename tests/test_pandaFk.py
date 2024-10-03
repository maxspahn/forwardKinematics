import os

import casadi as ca
import numpy as np
import pytest

from forwardkinematics import GenericURDFFk, GenericXMLFk
from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError


@pytest.fixture
def fk() -> GenericURDFFk:
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/panda.urdf"
    with open(urdf_file, "r") as file:
        urdf = file.read()
    fk_panda = GenericURDFFk(
        urdf,
        root_link="panda_link0",
        end_links="panda_leftfinger",
    )
    return fk_panda


@pytest.fixture
def fk_xml() -> GenericXMLFk:
    xml_file = os.path.dirname(os.path.abspath(__file__)) + "/panda.xml"
    with open(xml_file, "r") as file:
        xml = file.read()
    fk_panda = GenericXMLFk(
        xml,
        root_link="panda_link0",
        end_links="panda_leftfinger",
    )
    return fk_panda


def test_pandaFk(fk):
    q_ca = ca.SX.sym("q", 7)
    q_np = np.random.random(7)
    fkCasadi = fk.casadi(q_ca, "panda_link9", position_only=False)
    fkNumpy = fk.numpy(q_np, "panda_link9", position_only=False)
    assert isinstance(fkCasadi, ca.SX)
    assert isinstance(fkNumpy, np.ndarray)


def test_xmlFk(fk_xml):
    q_ca = ca.SX.sym("q", 9)
    fk_casadi = fk_xml.casadi(q_ca, "link7", position_only=False)
    assert isinstance(fk_casadi, ca.SX)


def test_compare_xml_urdf(fk, fk_xml):
    """
    Casadi expressions cannot be compared directly, as the expressions might
    be simplified differently but equivalent. Instead, we compare the size
    of the expressions and the values for a random q.
    """
    q_ca = ca.SX.sym("q", 9)
    for i in range(8):
        fk_casadi_urdf = fk.casadi(q_ca, f"panda_link{i}", position_only=False)
        fk_casadi_xml = fk_xml.casadi(q_ca, f"link{i}_c", position_only=False)
        print(fk_casadi_xml)
        assert fk_casadi_urdf.shape == fk_casadi_xml.shape
        fk_xml_function = ca.Function("fk_xml", [q_ca], [fk_casadi_xml])
        fk_urdf_function = ca.Function("fk_urdf", [q_ca], [fk_casadi_urdf])
        q_np = np.random.random(9)
        fk_np_xml = fk_xml_function(q_np)
        fk_np_urdf = fk_urdf_function(q_np)
        assert np.allclose(fk_np_xml, fk_np_urdf, atol=1e-4)
        fk_np_direct_xml = fk_xml.numpy(q_np, f"link{i}_c", position_only=False)
        assert np.allclose(fk_np_xml, fk_np_direct_xml, atol=1e-4)


def test_pandaFkByName(fk):
    q_ca = ca.SX.sym("q", 7)
    q_np = np.random.random(7)
    fkCasadi = fk.casadi(q_ca, "panda_link3", position_only=False)
    assert isinstance(fkCasadi, ca.SX)


def test_pandafkByWrongName(fk):
    q_ca = ca.SX.sym("q", 7)
    with pytest.raises(LinkNotInURDFError):
        fkCasadi = fk.casadi(q_ca, "panda_link10", position_only=False)


def test_simpleFk(fk: GenericURDFFk):
    q_np = np.array([0.0000, 1.0323, 0.0000, 0.8247, 0.0000, 0.2076, 0.0000])
    fkNumpy = fk.numpy(q_np, "panda_link0", position_only=True)
    x_ee = np.array([0.0, 0.0, 0.0])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.numpy(q_np, "panda_link3", position_only=True)
    x_ee = np.array([0.2713, 0.0, 0.4950])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)
    fkNumpy = fk.numpy(q_np, "panda_link9", position_only=True)
    x_ee = np.array([0.4, 0.0, 0.71])
    assert fkNumpy[0] == pytest.approx(x_ee[0], abs=1e-4)
    assert fkNumpy[1] == pytest.approx(x_ee[1], abs=1e-4)
    assert fkNumpy[2] == pytest.approx(x_ee[2], abs=1e-4)


def test_symbolic_fk(fk: GenericURDFFk):
    q_ca = ca.SX.sym("q", 7)
    panda_joint1_z = ca.SX.sym("panda_joint1_z")
    panda_joint3_roll = ca.SX.sym("panda_joint3_roll")
    symbolic_parameters = {
        "panda_joint1": {
            "z": panda_joint1_z,
        },
        "panda_joint3": {"roll": panda_joint3_roll},
    }

    link_name = "panda_link5"
    fkCasadi = fk.casadi(
        q_ca, link_name, position_only=False, symbolic_parameters=symbolic_parameters
    )
    assert isinstance(fkCasadi, ca.SX)
    fk_casadi_function = ca.Function(
        "fk", [q_ca, panda_joint1_z, panda_joint3_roll], [fkCasadi]
    )
    q_np = np.random.random(7)
    z = 0.333
    roll = 1.57079632679
    fk_np = fk.numpy(q_np, link_name, position_only=False)
    fk_symbolic_np = fk_casadi_function(q_np, z, roll)
    print(fk_np)
    print(fk_symbolic_np)
    assert np.allclose(fk_np, fk_symbolic_np, atol=1e-4)
