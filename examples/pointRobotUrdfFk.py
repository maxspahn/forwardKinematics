import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk


def main():
    with open("pointRobot.urdf", "r") as file:
        urdf = file.read()
    fk_point_robot = GenericURDFFk(
        urdf,
        rootLink = 'origin',
        end_link="base_link",
    )
    n = fk_point_robot.n()
    q_np = np.random.random(n)
    q_ca = ca.SX.sym("q", n)
    fkCasadi = fk_point_robot.casadi(q_ca, 'base_link', position_only=True)
    fkNumpy = fk_point_robot.numpy(q_np, 'base_link', position_only=True)
    print(fkNumpy)
    print(fkCasadi)

if __name__ == "__main__":
    main()
