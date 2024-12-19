import os

import casadi as ca
import numpy as np
import pytest

import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c


@pytest.fixture
def urdfparser() -> u2c.URDFparser:
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/panda.urdf"
    with open(urdf_file, "r") as file:
        urdf = file.read()
    robot = u2c.URDFparser(root_link="panda_link0", end_links=["panda_leftfinger"])
    robot.from_string(urdf)
    robot.detect_link_names()
    robot.set_joint_variable_map()
    return robot


def test_simple_link(urdfparser):
    assert isinstance(urdfparser, u2c.URDFparser)
    panda_joint1_x = ca.SX.sym("panda_joint1_x")
    panda_joint1_roll = ca.SX.sym("panda_joint1_roll")
    panda_joint1_yaw = ca.SX.sym("panda_joint1_yaw")
    symbolic_parameters = {
        "panda_joint1": {
            "x": panda_joint1_x,
            "roll": panda_joint1_roll,
            "yaw": panda_joint1_yaw,
        }
    }
    T = urdfparser.get_forward_kinematics(
        "panda_link0",
        "panda_link1",
        ca.vertcat([0, 0, 0, 0, 0, 0, 0]),
        symbolic_parameters=symbolic_parameters,
    )
    assert isinstance(T, dict)
    T_fk = T["T_fk"]
    print(T_fk)
    variable_names = [variable.name() for variable in ca.symvar(T_fk)]
    assert isinstance(T["T_fk"], ca.SX)
    assert "panda_joint1_x" in variable_names
    assert "panda_joint1_roll" in variable_names
    assert "panda_joint1_yaw" in variable_names
    assert T_fk[0, 3] == panda_joint1_x
