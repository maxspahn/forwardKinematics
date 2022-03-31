from abc import abstractmethod, ABC
import numpy as np
import casadi as ca


class ForwardKinematics(ABC):

    """Abstract class for forward kinematics"""

    def __init__(self):
        ABC.__init__(self)

    def fk(self, q, i, position_only=False):
        if isinstance(q, ca.SX):
            return self.casadi(q, i, position_only=position_only)
        elif isinstance(q, np.ndarray):
            return self.numpy(q, i, position_only=position_only)

    def n(self):
        return self._n

    @abstractmethod
    def casadi(self, q: ca.SX, i, position_only=False):
        pass

    @abstractmethod
    def numpy(self, q: np.ndarray, i, position_only=False):
        pass


