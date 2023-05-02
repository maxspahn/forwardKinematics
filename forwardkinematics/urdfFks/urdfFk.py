import os
import numpy as np
import casadi as ca
import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c

from forwardkinematics.fksCommon.fk import ForwardKinematics


class LinkNotInURDFError(Exception):
    pass


class URDFForwardKinematics(ForwardKinematics):
    def __init__(self, fileName, links, rootLink, end_links, n):
        super().__init__()
        self._urdf_file = (
            os.path.dirname(os.path.abspath(__file__)) + "/urdf/" + fileName
        )
        self._links = links
        self._rootLink = rootLink
        self._end_links = end_links
        self._q_ca = ca.SX.sym("q", n)
        self._mount_transformation = np.identity(4)
        self.readURDF()
        self._n = n

    def readURDF(self):
        self.robot = u2c.URDFparser(rootLink=self._rootLink, end_links=self._end_links)
        self.robot.from_file(self._urdf_file)
        self.robot.detect_link_names()
        self.robot.set_joint_variable_map()

    def generateFunctions(self):
        self._fks = {}
        for link in self.robot.link_names():
            ca_fun = ca.Function(
                "fk" + link, [self._q_ca], [self.casadi_by_name(self._q_ca, link)]
            )
            self._fks[link] = ca_fun

    def fk(self, q, link: str, positionOnly: bool = False):
        if isinstance(link, str):
            return self.fk_by_name(q, link, positionOnly=positionOnly)
        else:
            return super().fk(q, link, positionOnly=positionOnly)

    def fk_by_name(self, q: ca.SX, link: str, positionOnly=False):
        if isinstance(q, ca.SX):
            return self.casadi_by_name(q, link, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy_by_name(q, link, positionOnly=positionOnly)

    def casadi(self, q: ca.SX, i: int, positionOnly=False):
        return self.casadi_by_name(q, self._links[i], positionOnly=positionOnly)

    def casadi_by_name(self, q: ca.SX, link: str, positionOnly=False):
        """
        Raises:
            LinkNotInURDFError: An error occured accessing the urdf link.
        """
        if link not in self.robot.link_names():
            raise LinkNotInURDFError(
                f"""The link you have requested, {link}, is not in the urdf.
                    Possible links are  {self.robot.link_names()}"""
            )
        fk = self.robot.get_forward_kinematics(self._rootLink, link, q)["T_fk"]
        if positionOnly:
            return fk[0:3, 3]
        else:
            return fk

    def numpy(self, q: ca.SX, i: int, positionOnly=False):
        return self.numpy_by_name(q, self._links[i], positionOnly=positionOnly)

    def numpy_by_name(self, q: np.ndarray, link: str, positionOnly=False):
        if not hasattr(self, '_fks'):
            self.generateFunctions()
        """
        Raises:
            LinkNotInURDFError: An error occured accessing the urdf link.
        """
        if link not in self.robot.link_names():
            raise LinkNotInURDFError(
                f"""The link you have requested, {link}, is not in the urdf.
                    Possible links are  {self.robot.link_names()}"""
            )
        if positionOnly:
            return np.array(self._fks[link](q))[0:3, 3]
        else:
            return np.array(self._fks[link](q))
