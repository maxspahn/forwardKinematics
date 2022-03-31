import os
import numpy as np
import casadi as ca
import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c

from forwardkinematics.fksCommon.fk import ForwardKinematics

class URDFForwardKinematics(ForwardKinematics):
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
        self._fks = {}
        for link in self.robot.link_names():
            ca_fun = ca.Function("fk"+link, [self._q_ca], [self.casadi_by_name(self._q_ca, link)])
            self._fks[link] = ca_fun

    def fk_by_name(self, q: ca.SX, link: str, position_only=False):
        if isinstance(q, ca.SX):
            return self.casadiByName(q, link, position_only=position_only)
        elif isinstance(q, np.ndarray):
            return self.numpyByName(q, link, position_only=position_only)


    def casadi(self, q: ca.SX, i: int, position_only=False):
        return self.casadi_by_name(q, self._links[i], position_only=position_only)

    def casadi_by_name(self, q: ca.SX, link: str, position_only=False):
        if position_only:
            return self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"][
                0:3, 3
            ]
        else:
            return self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"]

    def numpy(self, q: ca.SX, i: int, position_only=False):
        return self.numpy_by_name(q, self._links[i], position_only=position_only)

    def numpy_by_name(self, q: np.ndarray, link: str, position_only=False):
        try:
            if position_only:
                return np.array(self._fks[link](q))[0:3, 3]
            else:
                return np.array(self._fks[link](q))
        except KeyError as e:
            print(f"Error {e}. No link with name {link} found in the list of functions. Returning identity.")
            if position_only:
                return np.zeros(3)
            else:
                return np.identity(4)
