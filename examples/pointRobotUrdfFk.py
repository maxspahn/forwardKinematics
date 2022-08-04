import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.pointRobotUrdfFk import PointRobotUrdfFk


def main():
    n = 3
    q_ca = ca.SX.sym("q", n)
    fk_point_robot = PointRobotUrdfFk()
    q_np = np.random.random(n)
    fkCasadi = fk_point_robot.fk(q_ca, 1, positionOnly=True)
    fkNumpy = fk_point_robot.fk(q_np, 3, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)

if __name__ == "__main__":
    main()
