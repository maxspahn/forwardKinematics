import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class TiagoFk(URDFForwardKinematics):

    def __init__(self, n):
        fileName = 'tiago.urdf'
        linkIndices = [1, 2, 3, 4, 5, 7]
        relevantLinks = ['base_link', 'torso_lift_link']
        relevantLinks += ['arm_left_' + str(i) + '_link' for i in linkIndices] + ['arm_left_tool_link']
        relevantLinks += ['arm_right_' + str(i) + '_link' for i in linkIndices] + ['arm_left_tool_link']
        relevantLinks += ['head_1_link', 'head_2_link']
        super().__init__(fileName, relevantLinks, 'base_link', n)


if __name__ == "__main__":
    n = 17
    q_ca = ca.SX.sym("q_ca", n)
    pfk = TiagoFk(n)
    fkCasadi = pfk.fk(q_ca, 15, positionOnly=False)
    q_np = np.zeros(n)
    q_np[13] = 0.1
    for i in range(n+1):
        fkNumpy = pfk.fk(q_np, i, positionOnly=True)
        print(fkNumpy)
