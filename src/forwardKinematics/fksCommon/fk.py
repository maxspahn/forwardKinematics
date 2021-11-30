from abc import abstractmethod, ABC
import os
import numpy as np
import casadi as ca


class ForwardKinematics(ABC):

    """Abstract class for forward kinematics"""

    def __init__(self):
        ABC.__init__(self)

    def fk(self, q, i, positionOnly=False):
        if isinstance(q, ca.SX):
            return self.casadi(q, i, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy(q, i, positionOnly=positionOnly)

    def n(self):
        return self._n

    @abstractmethod
    def casadi(self, q: ca.SX, i, positionOnly=False):
        pass

    @abstractmethod
    def numpy(self, q: np.ndarray, i, positionOnly=False):
        pass
