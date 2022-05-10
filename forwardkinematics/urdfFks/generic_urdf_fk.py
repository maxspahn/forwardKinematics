import numpy as np
import casadi as ca
import cmd
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics
import forwardkinematics.urdfFks.casadiConversion.urdfparser as u2c

class GenericURDFFk(URDFForwardKinematics):
    def __init__(self, fileName, rootLink='base_link'):
        self._urdf_file = fileName
        self._links = []
        self._rootLink = rootLink
        self.readURDF()
        self._n = len(self.robot.get_all_actuated_joints())
        self._q_ca = ca.SX.sym("q", self._n)
        self.generateFunctions()

    def readURDF(self):
        self.robot = u2c.URDFparser()
        self.robot.from_string(self._urdf_file)
        self.robot.detect_link_names()
        self.robot.set_joint_variable_map()


    def casadi_by_name(self, q: ca.SX, parent_link: str, child_link: str, positionOnly=False):
        if child_link not in self.robot.link_names():
            print(f"The link you have requested, {child_link}, is not in the urdf.")
            cli = cmd.Cmd()
            print("Possible links are")
            print("----")
            cli.columnize(self.robot.link_names(), displaywidth=10)
            print("----")
            return
        if positionOnly:
            return self.robot.get_forward_kinematics(parent_link, child_link, q)["T_fk"][
                0:3, 3
            ]
        else:
            return self.robot.get_forward_kinematics(parent_link, child_link, q)["T_fk"]

    def fk_by_name(self, q: ca.SX, parent_link: str, child_link: str, positionOnly=False):
        if isinstance(q, ca.SX):
            return self.casadi_by_name(q, parent_link, child_link, positionOnly=positionOnly)
        elif isinstance(q, np.ndarray):
            return self.numpy_by_name(q, child_link, positionOnly=positionOnly)

    def generateFunctions(self):
        print("Generating functions is not supported supported for generic fk.")

    def numpy(self, q: ca.SX, i: int, positionOnly=False):
        print("Not supported for generic fk.")

    def numpy_by_name(self, q: np.ndarray, link: str, positionOnly=False):
        print("Numpy fk is not supported for generic fk.")
