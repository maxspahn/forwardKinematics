from typing import Union
import numpy as np
import casadi as ca

class ForwardKinematics():

    """Abstract class for forward kinematics"""

    _n: int

    def __init__(self):
        pass

    def set_mount_transformation(self, mount_transformation: np.ndarray):
        self._mount_transformation = mount_transformation

    def n(self) -> int:
        return self._n

    def casadi(
            self, q: ca.SX,
            child_link: Union[int, str],
            parent_link: Union[int, str, None] = None,
            link_transformation=np.eye(4),
            position_only: bool = False
        ) -> ca.SX:
        raise NotImplementedError

    def numpy(
            self, q: np.ndarray,
            child_link: Union[int, str],
            parent_link: Union[int, str, None] = None,
            link_transformation=np.eye(4),
            position_only: bool = False
        ) -> np.ndarray:
        raise NotImplementedError


