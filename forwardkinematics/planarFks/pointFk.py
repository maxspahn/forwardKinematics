import numpy as np
import casadi as ca
from forwardkinematics.planarFks.planar_fk import ForwardKinematicsPlanar


class PointFk(ForwardKinematicsPlanar):
    def __init__(self):
        super().__init__()
        self._n = 2

    def fk(self, q, link, position_only: bool=False, endlink=0.0):
        if isinstance(link, str):
            i = self.get_link_index(link)
        else:
            i = link
        assert i <= self._n
        if isinstance(q, ca.SX):
            assert q.shape[0] == self._n
            return self.casadi(q, i, position_only=position_only, endlink=endlink)
        elif isinstance(q, np.ndarray):
            assert q.size == self._n
            return self.numpy(q, i, position_only=position_only, endlink=endlink)

    def casadi(self, q, i, position_only=False, endlink=0.0):
        fk = ca.SX(np.array([0.0, 0.0]))
        if i > 0:
            fk += q
        return fk

    def numpy(self, q, i, position_only=False, endlink=0.0):
        fk = np.array([0.0, 0.0])
        if i > 0:
            fk += q
        return fk
