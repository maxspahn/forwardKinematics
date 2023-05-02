from abc import abstractmethod, ABC
import numpy as np
import casadi as ca


class ForwardKinematics(ABC):

    """Abstract class for forward kinematics"""

    def __init__(self):
        ABC.__init__(self)
        self._n = None

    def set_mount_transformation(self, mount_transformation: np.ndarray):
        self._mount_transformation = mount_transformation

    def fk(self, q, link, positionOnly=False):
        if isinstance(q, ca.SX):
            return self.casadi(q, link, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy(q, link, positionOnly=positionOnly)

    def n(self):
        return self._n

    @abstractmethod
    def casadi(self, q: ca.SX, link, positionOnly=False):
        pass

    @abstractmethod
    def numpy(self, q: np.ndarray, link, positionOnly=False):
        pass


