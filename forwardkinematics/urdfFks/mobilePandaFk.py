import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class MobilePandaFk(URDFForwardKinematics):

    def __init__(self, n):
        fileName = 'mobilePanda.urdf'
        relevantLinks = ['world', 'base_link_y', 'base_link'] + ['panda_link' + str(i) for i in [0, 1, 3, 5, 6, 7, 8, 9]]
        super().__init__(fileName, relevantLinks, 'world', 10)
        print(relevantLinks)


if __name__ == "__main__":
    q_ca = ca.SX.sym("q_ca", 10)
    pfk = MobilePandaFk(10)
    fkCasadi = pfk.fk(q_ca, 10)
    q_np = np.array([-0.2, 0.6, 0.7, -0.01026582, -0.28908, 0.00922652, -1.3468, -0.026000, 2.074019, 0.06062268])
    fkNumpy = pfk.fk(q_np, 10, positionOnly=True)
    print(fkNumpy)
    for i in range(11):
        print(pfk.fk(q_np, i, positionOnly=True))
