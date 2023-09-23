from abc import abstractmethod, ABC
import numpy as np
import casadi as ca


class ForwardKinematicsByName(ABC):

    """Abstract class for forward kinematics"""

    def __init__(self):
        ABC.__init__(self)

    def fk(self, q, link: str, position_only: bool=False):
        if isinstance(q, ca.SX):
            return self.casadi(q, link, position_only=position_only)
        elif isinstance(q, np.ndarray):
            return self.numpy(q, link, position_only=position_only)

    def n(self):
        return self._n

    @abstractmethod
    def casadi(self, q: ca.SX, link: str, position_only=False) -> ca.SX:
        pass

    @abstractmethod
    def numpy(self, q: np.ndarray, link: str, position_only=False) -> np.ndarray:
        pass
