import os
import numpy as np
import cmd
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

    def fk_by_name(self, q: ca.SX, link: str, positionOnly=False):
        if isinstance(q, ca.SX):
            return self.casadi_by_name(q, link, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy_by_name(q, link, positionOnly=positionOnly)

    def casadi(self, q: ca.SX, i: int, positionOnly=False):
        return self.casadi_by_name(q, self._links[i], positionOnly=positionOnly)

    def casadi_by_name(self, q: ca.SX, link: str, positionOnly=False):
        if link not in self.robot.link_names():
            print(f"The link you have requested, {link}, is not in the urdf.")
            cli = cmd.Cmd()
            print("Possible links are")
            print("----")
            cli.columnize(self.robot.link_names(), displaywidth=10)
            print("----")
            return
        if positionOnly:
            return self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"][
                0:3, 3
            ]
        else:
            return self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"]

    def numpy(self, q: ca.SX, i: int, positionOnly=False):
        return self.numpy_by_name(q, self._links[i], positionOnly=positionOnly)

    def numpy_by_name(self, q: np.ndarray, link: str, positionOnly=False):
        if link not in self.robot.link_names():
            print(f"The link you have requested, {link}, is not in the urdf.")
            cli = cmd.Cmd()
            print("Possible links are")
            print("----")
            cli.columnize(self.robot.link_names(), displaywidth=10)
            print("----")
            return
        if positionOnly:
            return np.array(self._fks[link](q))[0:3, 3]
        else:
            return np.array(self._fks[link](q))
