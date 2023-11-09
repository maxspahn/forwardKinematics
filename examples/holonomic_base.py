import os
import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE=absolute_path + "/assets/pointRobot.urdf"


def main():
    with open(URDF_FILE, "r") as file:
        urdf = file.read()
    fk_point_robot = GenericURDFFk(
        urdf,
        root_link = 'origin',
        end_links="base_link",
    )
    dof = fk_point_robot.n()
    q_np = np.random.random(dof)
    q_ca = ca.SX.sym("q", dof)
    fk_casadi = fk_point_robot.casadi(q_ca, 'base_link', position_only=True)
    fk_numpy = fk_point_robot.numpy(q_np, 'base_link', position_only=True)
    return fk_casadi, fk_numpy

if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_numpy)
    print(fk_casadi)
