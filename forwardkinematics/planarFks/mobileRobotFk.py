import numpy as np
import casadi as ca
from forwardkinematics.fksCommon.fk import ForwardKinematics


class MobileRobotFk(ForwardKinematics):
    def __init__(self, n, baseHeight=1.0):
        super().__init__()
        self._n = n
        self._baseHeight = baseHeight

    def fk(self, q, i, position_only=False, endlink=0.0):
        assert i <= self._n
        if isinstance(q, ca.SX):
            assert q.shape[0] == self._n
            return self.casadi(q, i, position_only=position_only, endlink=endlink)
        elif isinstance(q, np.ndarray):
            assert q.size == self._n
            return self.numpy(q, i, position_only=position_only, endlink=endlink)

    def casadi(self, q, i, position_only=False, endlink=0.0):
        fk = ca.SX(np.zeros(3))
        if i > 0:
            if self._n > 1:
                fk = ca.SX(np.array([q[0], self._baseHeight + 0.2, q[1]]))
            else:
                fk = ca.SX(np.array([q[0], self._baseHeight + 0.2, 0.0]))
        for i in range(2, i + 1):
            fk[0] += ca.cos(fk[2]) * 1.0
            fk[1] += ca.sin(fk[2]) * 1.0
            if i < q.size(1):
                fk[2] += q[i]
        fk[0] += ca.cos(fk[2]) * endlink
        fk[1] += ca.sin(fk[2]) * endlink
        if position_only:
            return fk[0:2]
        else:
            return fk

    def numpy(self, q, i, position_only=False, endlink=0.0):
        fk = np.zeros(3)
        if i > 0:
            if self._n > 1:
                fk = np.array([q[0], self._baseHeight + 0.2, q[1]])
            else:
                fk = np.array([q[0], self._baseHeight + 0.2, 0.0])
        for i in range(2, i + 1):
            fk[0] += np.cos(fk[2]) * 1.0
            fk[1] += np.sin(fk[2]) * 1.0
            if i < len(q):
                fk[2] += q[i]
        fk[0] += np.cos(fk[2]) * endlink
        fk[1] += np.sin(fk[2]) * endlink
        if position_only:
            return fk[0:2]
        else:
            return fk
