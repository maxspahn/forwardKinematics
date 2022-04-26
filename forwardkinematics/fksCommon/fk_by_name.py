from abc import abstractmethod, ABC
import numpy as np
import casadi as ca


class ForwardKinematicsByName(ABC):

    """Abstract class for forward kinematics"""

    def __init__(self):
        ABC.__init__(self)

    def fk(self, q, link: str, positionOnly: bool=False):
        if isinstance(q, ca.SX):
            return self.casadi(q, link, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy(q, link, positionOnly=positionOnly)

    def n(self):
        return self._n

    @abstractmethod
    def casadi(self, q: ca.SX, link: str, positionOnly=False) -> ca.SX:
        pass

    @abstractmethod
    def numpy(self, q: np.ndarray, link: str, positionOnly=False) -> np.ndarray:
        pass
