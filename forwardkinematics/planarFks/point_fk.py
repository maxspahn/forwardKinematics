from typing import Union

import numpy as np
import casadi as ca

from forwardkinematics.planarFks.planar_fk import ForwardKinematicsPlanar


class PointFk(ForwardKinematicsPlanar):
    def __init__(self):
        super().__init__()
        self._n = 2

    def casadi(
        self,
        q: ca.SX,
        child_link: Union[int, str],
        parent_link: Union[int, str, None] = None,
        link_transformation=np.eye(3),
        position_only: bool = False,
    ):
        fk = ca.SX.eye(3)
        child_link = self.ensure_int_link(child_link)
        if not parent_link:
            parent_link = 0
        parent_link = self.ensure_int_link(parent_link)
        if not parent_link:
            parent_link = 0
        if parent_link == 0 and child_link > 0:
            fk[0:2, 2] = q
        fk = ca.mtimes(fk, link_transformation)
        if position_only:
            return fk[0:2, 2]
        else:
            return fk

    def numpy(
        self,
        q: np.ndarray,
        child_link: Union[int, str],
        parent_link: Union[int, str, None] = None,
        link_transformation=np.eye(3),
        position_only: bool = False,
    ):
        fk = np.eye(3)
        child_link = self.ensure_int_link(child_link)
        if not parent_link:
            parent_link = 0
        parent_link = self.ensure_int_link(parent_link)
        if not parent_link:
            parent_link = 0
        if parent_link == 0 and child_link > 0:
            fk[0:2, 2] = q
        fk = np.dot(fk, link_transformation)
        if position_only:
            return fk[0:2, 2]
        else:
            return fk
