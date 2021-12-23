import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class BoxerFk(URDFForwardKinematics):

    def __init__(self, n):
        fileName = 'boxer.urdf'
        relevantLinks = ['world', 'ee_link', 'ee_link', 'ee_link']
        super().__init__(fileName, relevantLinks, 'world', 3)


if __name__ == "__main__":
    q_ca = ca.SX.sym("q_ca", 3)
    bfk = BoxerFk(3)
    fkCasadi = bfk.fk(q_ca, 0, positionOnly=True)
    print(fkCasadi)
    q_np = np.array([0.0, 1.0, 1 * np.pi/2.0])
    for i in range(bfk.n()):
        print(bfk.fk(q_np, i, positionOnly=True))
