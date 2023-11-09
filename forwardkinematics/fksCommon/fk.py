import numpy as np

class ForwardKinematics():

    """Abstract class for forward kinematics"""

    _n: int

    def __init__(self):
        pass

    def set_mount_transformation(self, mount_transformation: np.ndarray):
        self._mount_transformation = mount_transformation

    def n(self) -> int:
        return self._n


