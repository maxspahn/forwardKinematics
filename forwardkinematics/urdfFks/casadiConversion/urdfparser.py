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
    
    def __init__(self, rootLink: str="base_link", func_opts=None):
        #self.robot_desc = None
        self._rootLink = rootLink
        if func_opts:
            self.func_opts = func_opts

    def from_file(self, filename):
        """Uses an URDF file to get robot description."""
        self.robot_desc = URDF.from_xml_file(filename)
        self.detect_link_names()
        self._absolute_root_link = self.robot_desc.get_root()
        self.set_active_joints()

    def from_server(self, key="robot_description"):
        """Uses a parameter server to get robot description."""
        self.robot_desc = URDF.from_parameter_server(key=key)
        self._absolute_root_link = self.robot_desc.get_root()
        self.set_active_joints()

    def from_string(self, urdfstring):
        """Uses a URDF string to get robot description."""
        self.robot_desc = URDF.from_xml_string(urdfstring)
        self._absolute_root_link = self.robot_desc.get_root()
        self.set_active_joints()

    def get_all_actuated_joints(self) -> list:
        actuated_joints_names = []
        for joint in self.robot_desc.joints:
            if joint.type in self.actuated_types:
                actuated_joints_names.append(joint.name)
        return actuated_joints_names

    def set_joint_variable_map(self) -> None:
        joint_names = self.get_active_joints()
        actuated_joints = self.get_all_actuated_joints()
        self._joint_map = {}
        index = 0
        for joint_name in joint_names:
            if joint_name in actuated_joints:
                self._joint_map[joint_name] = index
                index += 1

    def is_active_joint(self, joint):
        parent_link = joint.parent
        while not (parent_link == self._rootLink or parent_link == self._absolute_root_link):
            parent_link = self.robot_desc.parent_map[parent_link][1]
        if parent_link == self._rootLink:
            return True

    def set_active_joints(self) -> None:
        self._active_joints = []
        actuated_joints = self.get_all_actuated_joints()
        for joint in self.robot_desc.joints:
            if self.is_active_joint(joint):
                self._active_joints.append(joint.name)

    def get_active_joints(self) -> list:
        return self._active_joints
        

    def get_joint_info(self, root, tip):
        """Using an URDF to extract joint information, i.e list of
        joints, actuated names and upper and lower limits."""
        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        joint_list = []
        upper = []
        lower = []
        actuated_names = self.get_all_actuated_joints()

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.name in self._active_joints:
                    joint_list += [joint]
                    if joint.name in actuated_names:
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

    def link_names(self):
        return self._link_names

    def detect_link_names(self):
        self._link_names = []
        for link in self.robot_desc.links:
            if link.name in self.robot_desc.parent_map:
                self._link_names.append(link.name)
            else:
                print(f"Link with name {link.name} does not has a parent. Link name is skipped.")
        return self._link_names


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
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        joint_list, actuated_names, upper, lower = self.get_joint_info(
            self._absolute_root_link,
            tip)
        T_fk = cs.SX.eye(4)
        # nvar = len(actuated_names)
        # q = cs.SX.sym("q", nvar)
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
