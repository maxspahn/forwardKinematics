import numpy as np
import casadi as ca
from forwardkinematics.planarFks.planar_fk import ForwardKinematicsPlanar


class PlanarArmFk(ForwardKinematicsPlanar):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def casadi(self, q, link, positionOnly=False, endlink=0.0):
        fk = ca.vertcat(np.zeros(2), q[0])
        for i in range(1, link + 1):
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

    def numpy(self, q, link, positionOnly=False, endlink=0.0):
        fk = np.array([0.0, 0.0, q[0]])
        for i in range(1, link + 1):
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
