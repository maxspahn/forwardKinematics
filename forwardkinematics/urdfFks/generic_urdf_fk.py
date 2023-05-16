import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics, LinkNotInURDFError
import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c

class GenericURDFFk(URDFForwardKinematics):
    def __init__(self, fileName, rootLink='base_link', end_link=None, base_type: str = 'holonomic'):
        self._urdf_file = fileName
        self._links = []
        self._end_link = end_link
        self._rootLink = rootLink
        self.readURDF(rootLink, end_link)
        self._n = self.robot.degrees_of_freedom()
        if base_type in ['diffdrive']:
            self._q_base = ca.SX.sym("q_base", 3)
        self._base_type = base_type
        self._q_ca = ca.SX.sym("q", self._n)
        self._mount_transformation = np.identity(4)

    def readURDF(self, rootLink: str, end_link: str):
        self.robot = u2c.URDFparser(rootLink, end_link)
        self.robot.from_string(self._urdf_file)
        self.robot.detect_link_names()
        self.robot.set_joint_variable_map()

    def generateFunctions(self):
        self._fks = {}
        for link in self.robot.link_names():
            if self._base_type in ['diffdrive']:
                q = ca.vcat([self._q_base, self._q_ca])
            else:
                q = self._q_ca
            ca_fun = ca.Function(
                "fk" + link, [q], [self.casadi(q, self._rootLink, link)]
            )
            self._fks[link] = ca_fun


    def casadi(self, q: ca.SX, parent_link: str, child_link: str, link_transformation=np.eye(4), positionOnly=False):
        if child_link not in self.robot.link_names():
            raise LinkNotInURDFError(
                f"""The link you have requested, {child_link}, is not in the urdf.
                    Possible links are  {self.robot.link_names()}"""
            )
        if self._base_type in ['diffdrive']:
            fk = self.robot.get_forward_kinematics(parent_link, child_link, q[2:], link_transformation)["T_fk"]
            c = ca.cos(q[2])
            s = ca.sin(q[2])
            T_base = ca.vcat([
                ca.hcat([c, -s, 0, q[0]]),
                ca.hcat([s, c, 0, q[1]]),
                ca.hcat([0, 0, 1, 0]),
                ca.hcat([0, 0, 0, 1]),
            ])
            fk = ca.mtimes(T_base, fk)
        else:
            fk = self.robot.get_forward_kinematics(parent_link, child_link, q, link_transformation)["T_fk"]
            fk = ca.mtimes(self._mount_transformation, fk)

        if positionOnly:
            fk = fk[0:3, 3]
        return fk

    def fk(self, q: ca.SX, parent_link: str, child_link: str,link_transformation=np.eye(4), positionOnly=False):
        if isinstance(q, ca.SX):
            return self.casadi(q, parent_link, child_link, link_transformation, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy(q, parent_link, child_link, link_transformation, positionOnly=positionOnly)

    def numpy(self, q: ca.SX, parent_link: str, child_link: str, link_transformation=np.eye(4), positionOnly=False):
        if parent_link == self._rootLink:
            fk_parent = np.identity(4)
        else:
            fk_parent = super().numpy_by_name(q, parent_link)
        fk_child = super().numpy_by_name(q, child_link)
        tf_parent_child = np.dot(np.linalg.inv(fk_parent), fk_child)
        tf_parent_child = np.dot(link_transformation, tf_parent_child) #ToDo check if correct
        if positionOnly:
            return tf_parent_child[0:3, 3]
        else:
            return tf_parent_child

