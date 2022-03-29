from abc import abstractmethod, ABC
import os
import numpy as np
import casadi as ca
import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c

from forwardkinematics.fksCommon.fk_by_name import ForwardKinematicsByName


class URDFForwardKinematics(ForwardKinematicsByName):
    def __init__(self, fileName, links, rootLink, n):
        super().__init__()
        self._urdf_file = (
            os.path.dirname(os.path.abspath(__file__)) + "/urdf/" + fileName
        )
        self._links = links
        self._rootLink = rootLink
        self._q_ca = ca.SX.sym("q", n)
        self.readURDF()
        self._n = n
        self.generateFunctions()

    def readURDF(self):
        self.robot = u2c.URDFparser()
        self.robot.from_file(self._urdf_file)
        self.robot.set_joint_variable_map()

    def generateFunctions(self):
        fks = []
        for i in range(self._n + 1):
            fks.append(self.casadi(self._q_ca, i))
        self._fks_fun = ca.Function("fk", [self._q_ca], fks)

    def casadi(self, q: ca.SX, link: str, positionOnly=False):
        if positionOnly:
            return self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"][
                0:3, 3
            ]
        else:
            return self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"]

    def numpy(self, q: np.ndarray, link, positionOnly=False):
        # TODO: If the function is composed at runtime, this will be very slow. Therefore, the fk functions should be generated for all possible links. Then at runtime, the right function is accessed. This could be solved using a dict.<29-03-22, mspahn> #
        i = 3 # replaced with a mapping i = self._linkMap[link]
        if positionOnly:
            return np.array(self._fks_fun(q)[i])[0:3, 3]
        else:
            return np.array(self._fks_fun(q)[i])
