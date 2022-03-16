import numpy as np
import casadi as ca
from forwardkinematics.fksCommon.fk import ForwardKinematics


class PointFk(ForwardKinematics):
    def __init__(self):
        super().__init__()
        self._n = 2

    def fk(self, q, i, positionOnly=False, endlink=0.0):
        assert i <= self._n
        if isinstance(q, ca.SX):
            assert q.shape[0] == self._n
            return self.casadi(q, i, positionOnly=positionOnly, endlink=endlink)
        elif isinstance(q, np.ndarray):
            assert q.size == self._n
            return self.numpy(q, i, positionOnly=positionOnly, endlink=endlink)

    def casadi(self, q, i, positionOnly=False, endlink=0.0):
        fk = ca.SX(np.array([0.0, 0.0]))
        if i > 0:
            fk += q
        return fk

    def numpy(self, q, i, positionOnly=False, endlink=0.0):
        fk = np.array([0.0, 0.0])
        if i > 0:
            fk += q
        return fk
