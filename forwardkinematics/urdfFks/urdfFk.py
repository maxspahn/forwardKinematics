import os
from typing import Union
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
        self.read_urdf()
        self._n = n

    def n(self) -> int:
        return self._n

    def read_urdf(self):
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

    def fk(self, q, link: str, position_only: bool = False):
        return self.fk_by_name(q, link, position_only=position_only)

    def fk_by_name(self, q: Union[ca.SX, np.ndarray], link: str, position_only=False):
        if isinstance(q, ca.SX):
            return self.casadi_by_name(q, link, position_only=position_only)
        if isinstance(q, np.ndarray):
            return self.numpy_by_name(q, link, position_only=position_only)

    def casadi_by_name(self, q: ca.SX, link: str, position_only=False):
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
        if position_only:
            return fk[0:3, 3]
        else:
            return fk

    def numpy_by_name(self, q: np.ndarray, link: str, position_only=False):
        """
        Raises:
            LinkNotInURDFError: An error occured accessing the urdf link.
        """
        if not hasattr(self, '_fks'):
            self.generateFunctions()
        if link == self._rootLink:
            fk = np.identity(4)
        else:
            if link not in self.robot.link_names():
                raise LinkNotInURDFError(
                    f"""The link you have requested, {link}, is not in the urdf.
                        Possible links are  {self.robot.link_names()}"""
                )
            fk = np.array(self._fks[link](q))
        if position_only:
            return fk[0:3, 3]
        else:
            return fk
