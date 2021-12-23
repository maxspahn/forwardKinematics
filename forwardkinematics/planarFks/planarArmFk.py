import numpy as np
import casadi as ca
from forwardkinematics.fksCommon.fk import ForwardKinematics


class PlanarArmFk(ForwardKinematics):

    def __init__(self, n):
        super().__init__()
        self._n = n

    def fk(self, q, i, positionOnly=False, endlink=0.0):
        assert i <= self._n
        if isinstance(q, ca.SX):
            assert(q.shape[0] == self._n)
            return self.casadi(q, i, positionOnly=positionOnly, endlink=endlink)
        elif isinstance(q, np.ndarray):
            assert(q.size == self._n)
            return self.numpy(q, i, positionOnly=positionOnly, endlink=endlink)

    def casadi(self, q, i, positionOnly=False, endlink=0.0):
        fk = ca.SX(np.array([0.0, 0.0, q[0]]))
        for i in range(1, i + 1):
            fk[0] += ca.cos(fk[2]) * 1.0
            fk[1] += ca.sin(fk[2]) * 1.0
            if i < q.size(1):
                fk[2] += q[i]
        fk[0] += ca.cos(fk[2]) * endlink
        fk[1] += ca.sin(fk[2]) * endlink
        if positionOnly:
            return fk[0:2]
        else:
            return fk

    def numpy(self, q, i, positionOnly=False, endlink=0.0):
        fk = np.array([0.0, 0.0, q[0]])
        for i in range(1, i+1):
            fk[0] += np.cos(fk[2]) * 1.0
            fk[1] += np.sin(fk[2]) * 1.0
            if i < len(q):
                fk[2] += q[i]
        fk[0] += np.cos(fk[2]) * endlink
        fk[1] += np.sin(fk[2]) * endlink
        if positionOnly:
            return fk[0:2]
        else:
            return fk


if __name__ == "__main__":
    q_ca = ca.SX.sym("q", 6)
    fkPlanar = PlanarArmFk(6)
    q_np = np.array([3.0, 0.0, 0.2, 0.0, 0.0, 0.2])
    fkCasadi = fkPlanar.fk(q_ca, 6, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 6, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)

