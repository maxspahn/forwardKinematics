from typing import Union
import numpy as np
import casadi as ca
from forwardkinematics.planarFks.planar_fk import ForwardKinematicsPlanar


def get_rotation_matrix_casadi(angle: ca.SX) -> ca.SX:
    print(angle)
    return ca.SX(
        ca.vcat(
            [
                ca.hcat([ca.cos(angle), -ca.sin(angle), 0]),
                ca.hcat([ca.sin(angle), ca.cos(angle), 0]),
                ca.hcat([0, 0, 1]),
            ]
        )
    )


def get_rotation_matrix_numpy(angle: ca.SX) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


class PlanarArmFk(ForwardKinematicsPlanar):
    def __init__(self, n):
        super().__init__()
        self._n = n

    def casadi(
        self,
        q: ca.SX,
        child_link: Union[int, str],
        parent_link: Union[int, str, None] = None,
        link_transformation=np.eye(3),
        position_only: bool = False,
    ):
        child_link = self.ensure_int_link(child_link)
        if not parent_link:
            parent_link = 0
        parent_link = self.ensure_int_link(parent_link)
        if not parent_link:
            parent_link = 0
        fk = get_rotation_matrix_casadi(q[parent_link])
        for i in range(parent_link + 1, child_link + 1):
            if i == self.n():
                fk_i = get_rotation_matrix_casadi(0)
            else:
                fk_i = get_rotation_matrix_casadi(q[i])
            fk_i[0, 2] = 1
            fk = ca.mtimes(fk, fk_i)
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
        child_link = self.ensure_int_link(child_link)
        if not parent_link:
            parent_link = 0
        parent_link = self.ensure_int_link(parent_link)
        fk = get_rotation_matrix_numpy(q[parent_link])
        for i in range(parent_link + 1, child_link + 1):
            if i == self.n():
                fk_i = get_rotation_matrix_numpy(0)
            else:
                fk_i = get_rotation_matrix_numpy(q[i])
            fk_i[0, 2] = 1
            fk = np.dot(fk, fk_i)
        fk = np.dot(fk, link_transformation)
        if position_only:
            return fk[0:2, 2]
        else:
            return fk
