from abc import abstractmethod, ABC
import os
import numpy as np
import casadi as ca
import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c

from forwardkinematics.fksCommon.fk import ForwardKinematics

class URDFForwardKinematics(ForwardKinematics):

    def __init__(self, fileName, links, rootLink, n):
        super().__init__()
        self._urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/urdf/" + fileName
        self._links = links
        self._rootLink = rootLink
        self.readURDF()
        self._n = n
        self._q_ca = ca.SX.sym('q', n)
        self.generateFunctions()

    def readURDF(self):
        self.robot = u2c.URDFparser()
        self.robot.from_file(self._urdf_file)

    def generateFunctions(self):
        fks = []
        for i in range(self._n + 1):
            fks.append(self.casadi(self._q_ca, i))
        self._fks_fun = ca.Function('fk', [self._q_ca], fks)

    def casadi(self, q: ca.SX, i, positionOnly=False):
        tip = self._links[i]
        if positionOnly:
            return self.robot.get_forward_kinematics(self._rootLink, tip, q)['T_fk'][0:3, 3]
        else:
            return self.robot.get_forward_kinematics(self._rootLink, tip, q)['T_fk']

    def numpy(self, q: np.ndarray, i, positionOnly=False):
        if positionOnly:
            return np.array(self._fks_fun(q)[i])[0:3, 3]
        else:
            return np.array(self._fks_fun(q)[i])
