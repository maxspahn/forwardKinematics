"""This module is in most parts copied from https://github.com/mahaarbo/urdf2casadi.

Changes are in get_forward_kinematics as it allows to pass the variable as an argument.
"""
import casadi as cs
import numpy as np
from platform import machine, system
from urdf_parser_py.urdf import URDF, Pose
import forwardkinematics.urdfFks.casadiConversion.geometry.transformation_matrix as T


class URDFparser(object):
    """Class that turns a chain from URDF to casadi functions."""
    actuated_types = ["prismatic", "revolute", "continuous"]
    func_opts = {"jit": True, "jit_options": {"flags": "-Ofast"}}
    # OS/CPU dependent specification of compiler
    if system() == "darwin" or machine() == "aarch64":
        func_opts["compiler"] = "shell"
    
    def __init__(self, func_opts=None):
        self.robot_desc = None
        if func_opts:
            self.func_opts = func_opts

    def from_file(self, filename):
        """Uses an URDF file to get robot description."""
        self.robot_desc = URDF.from_xml_file(filename)

    def from_server(self, key="robot_description"):
        """Uses a parameter server to get robot description."""
        self.robot_desc = URDF.from_parameter_server(key=key)

    def from_string(self, urdfstring):
        """Uses a URDF string to get robot description."""
        self.robot_desc = URDF.from_xml_string(urdfstring)

    def get_all_actuated_joints(self) -> list:
        actuated_joints_names = []
        for joint in self.robot_desc.joints:
            if joint.type in self.actuated_types:
                actuated_joints_names.append(joint.name)
        return actuated_joints_names

    def set_joint_variable_map(self) -> None:
        joint_names = self.get_all_actuated_joints()
        self._joint_map = {}
        for i, joint_name in enumerate(joint_names):
            self._joint_map[joint_name] = i
        

    def get_joint_info(self, root, tip):
        """Using an URDF to extract joint information, i.e list of
        joints, actuated names and upper and lower limits."""
        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        joint_list = []
        upper = []
        lower = []
        actuated_names = []

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                joint_list += [joint]
                if joint.type in self.actuated_types:
                    actuated_names += [joint.name]
                    if joint.type == "continuous":
                        upper += [cs.inf]
                        lower += [-cs.inf]
                    else:
                        upper += [joint.limit.upper]
                        lower += [joint.limit.lower]
                    if joint.axis is None:
                        joint.axis = [1., 0., 0.]
                    if joint.origin is None:
                        joint.origin = Pose(xyz=[0., 0., 0.],
                                            rpy=[0., 0., 0.])
                    elif joint.origin.xyz is None:
                        joint.origin.xyz = [0., 0., 0.]
                    elif joint.origin.rpy is None:
                        joint.origin.rpy = [0., 0., 0.]

        return joint_list, actuated_names, upper, lower


    def get_n_joints(self, root, tip):
        """Returns number of actuated joints."""

        chain = self.robot_desc.get_chain(root, tip)
        n_actuated = 0

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type in self.actuated_types:
                    n_actuated += 1

        return n_actuated

    def get_forward_kinematics(self, root, tip, q):
        """Returns the forward kinematics as a casadi function."""
        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        joint_list, actuated_names, upper, lower = self.get_joint_info(
            root,
            tip)
        T_fk = cs.SX.eye(4)
        # nvar = len(actuated_names)
        # q = cs.SX.sym("q", nvar)
        i = 0
        for joint in joint_list:
            if joint.type == "fixed":
                xyz = joint.origin.xyz
                rpy = joint.origin.rpy
                joint_frame = T.numpy_rpy(xyz, *rpy)
                T_fk = cs.mtimes(T_fk, joint_frame)

            elif joint.type == "prismatic":
                if joint.axis is None:
                    axis = cs.np.array([1., 0., 0.])
                else:
                    axis = cs.np.array(joint.axis)
                # axis = (1./cs.np.linalg.norm(axis))*axis
                joint_frame = T.prismatic(joint.origin.xyz,
                                          joint.origin.rpy,
                                          joint.axis, q[self._joint_map[joint.name]])
                T_fk = cs.mtimes(T_fk, joint_frame)

            elif joint.type in ["revolute", "continuous"]:
                if joint.axis is None:
                    axis = cs.np.array([1., 0., 0.])
                else:
                    axis = cs.np.array(joint.axis)
                axis = (1./cs.np.linalg.norm(axis))*axis
                joint_frame = T.revolute(
                    joint.origin.xyz,
                    joint.origin.rpy,
                    joint.axis, q[self._joint_map[joint.name]])
                T_fk = cs.mtimes(T_fk, joint_frame)
        #T_fk = cs.Function("T_fk", [q], [T_fk], self.func_opts)
        return {
            "T_fk": T_fk
        }
