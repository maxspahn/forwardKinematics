import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class PandaFk(URDFForwardKinematics):

    def __init__(self, n):
        fileName = 'panda.urdf'
        relevantLinks = ['panda_link' + str(i) for i in [0, 3, 4, 5, 6, 7, 8, 9]]
        super().__init__(fileName, relevantLinks, 'panda_link0', 7)


if __name__ == "__main__":
    q_ca = ca.SX.sym("q_ca", 7)
    pfk = PandaFk(7)
    fkCasadi = pfk.fk(q_ca, 7)
    q_np = np.array([-0.01026582, -0.28908, 0.00922652, -1.3468, -0.026000, 2.074019, 0.06062268])
    fkNumpy = pfk.fk(q_np, 7, positionOnly=True)
    print(fkNumpy)
