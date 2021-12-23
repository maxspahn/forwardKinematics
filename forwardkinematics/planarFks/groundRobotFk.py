import numpy as np
import casadi as ca
from forwardkinematics.fksCommon.fk import ForwardKinematics


class GroundRobotFk(ForwardKinematics):

    def __init__(self, n=3):
        super().__init__()
        self._n = n
        self._ee_link = 0.2

    def fk(self, q, i, positionOnly=False, endlink=0.0):
        assert i <= self._n
        if isinstance(q, ca.SX):
            #assert(q.shape[0] == self._n)
            return self.casadi(q, i, positionOnly=positionOnly, endlink=endlink)
        elif isinstance(q, np.ndarray):
            #assert(q.size == self._n)
            return self.numpy(q, i, positionOnly=positionOnly, endlink=endlink)

    def casadi(self, q, i, positionOnly=False, endlink=0.0):
        fk = ca.SX(np.array([0.0, 0.0, 0.0]))
        if i > 0:
            fk += q[0:3]
        if i > 1:
            fk[0:2] += self._ee_link * ca.vertcat(ca.cos(fk[2]), ca.sin(fk[2]))
        if i > 2 and i < self.n():
            fk[2] += q[i]
            fk[0:2] += ca.vertcat(ca.cos(fk[2]), ca.sin(fk[2]))
        if positionOnly:
            return fk[0:2]
        else:
            return fk

    def numpy(self, q, i, positionOnly=False, endlink=0.0):
        fk = np.array([0.0, 0.0, 0.0])
        if i > 0:
            fk += q[0:3]
            fk[0:2] += self._ee_link * np.array([np.cos(fk[2]), np.sin(fk[2])])
        if i > 2 and i < self.n():
            fk[2] += q[i]
            fk[0:2] += np.array([np.cos(fk[2]), np.sin(fk[2])])
        if positionOnly:
            return fk[0:2]
        else:
            return fk


if __name__ == "__main__":
    q_ca = ca.SX.sym("q", 4)
    fkPlanar = GroundRobotFk(4)
    q_np = np.array([0.0, 0.0, 0.0, 1.57])
    fkCasadi = fkPlanar.fk(q_ca, 1, positionOnly=True)
    print(fkCasadi)
    fkNumpy = fkPlanar.fk(q_np, 1, positionOnly=True)
    print(fkNumpy)

