"""This module is in most parts copied from https://github.com/mahaarbo/urdf2casadi.

Changes are in get_forward_kinematics as it allows to pass the variable as an argument.
"""
import casadi as ca
import numpy as np
from urdf_parser_py.urdf import URDF
import forwardkinematics.urdfFks.casadiConversion.geometry.transformation_matrix as T


class URDFparser(object):
    """Class that turns a chain from URDF to casadi functions."""
    actuated_types = ["prismatic", "revolute", "continuous"]
    
    def __init__(self, rootLink: str="base_link", end_links: list = None):
        self._rootLink = rootLink
        if isinstance(end_links, str):
            self._end_links = [end_links]
        else:
            self._end_links = end_links

    def extract_information(self):
        self._actuated_joints = []
        self._active_joints = set()
        self._degrees_of_freedom = 0
        self.detect_link_names()
        self._absolute_root_link = self.robot_desc.get_root()
        self.set_active_joints()
        self.set_actuated_joints()
        self.extract_degrees_of_freedom()

    def from_file(self, filename):
        """Uses an URDF file to get robot description."""
        self.robot_desc = URDF.from_xml_file(filename)
        self.extract_information()

    def from_server(self, key="robot_description"):
        """Uses a parameter server to get robot description."""
        self.robot_desc = URDF.from_parameter_server(key=key)
        self.extract_information()

    def from_string(self, urdfstring):
        """Uses a URDF string to get robot description."""
        self.robot_desc = URDF.from_xml_string(urdfstring)
        self.extract_information()

    def set_actuated_joints(self) -> None:
        for joint in self.robot_desc.joints:
            if joint.type in self.actuated_types:
                self._actuated_joints.append(joint.name)

    def degrees_of_freedom(self) -> int:
        return self._degrees_of_freedom

    def extract_degrees_of_freedom(self) -> None:
        self._degrees_of_freedom = 0
        for joint_name in self._active_joints:
            if joint_name in self._actuated_joints:
                self._degrees_of_freedom += 1

    def actuated_joints(self) -> list:
        return self._actuated_joints

    def set_joint_variable_map(self) -> None:
        self._joint_map = {}
        index = 0
        for joint_name in self._actuated_joints:
            if joint_name in self._active_joints:
                self._joint_map[joint_name] = index
                index += 1

    def is_active_joint(self, joint):
        parent_link = joint.parent
        while parent_link not in [self._rootLink, self._absolute_root_link]:
            if parent_link in self._end_links:
                return False
            parent_joint, parent_link = self.robot_desc.parent_map[parent_link]
            if parent_joint in self._active_joints:
                return True

        if parent_link == self._rootLink:
            return True
        return False

    def set_active_joints(self) -> None:
        for parent_link in self._end_links:
            while parent_link not in [self._rootLink, self._absolute_root_link]:
                parent_joint, parent_link = self.robot_desc.parent_map[parent_link]
                self._active_joints.add(parent_joint)
                if parent_link == self._rootLink:
                    break
        
    def active_joints(self) -> set:
        return self._active_joints
        

    def get_joint_info(self, root, tip) -> list:
        """Using an URDF to extract joint information, i.e list of
        joints, actuated names and upper and lower limits."""
        chain = self.robot_desc.get_chain(root, tip)
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')

        joint_list = []

        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.name in self._active_joints:
                    joint_list += [joint]

        return joint_list

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

    def get_forward_kinematics(self, root, tip, q, link_transformation=np.eye(4)):
        """Returns the forward kinematics as a casadi function."""
        if self.robot_desc is None:
            raise ValueError('Robot description not loaded from urdf')
        joint_list = self.get_joint_info(self._absolute_root_link, tip)
        T_fk = ca.SX.eye(4)
        for joint in joint_list:
            if joint.type == "fixed":
                xyz = joint.origin.xyz
                rpy = joint.origin.rpy
                joint_frame = T.numpy_rpy(xyz, *rpy)
                T_fk = ca.mtimes(T_fk, joint_frame)

            elif joint.type == "prismatic":
                if joint.axis is None:
                    axis = ca.np.array([1., 0., 0.])
                else:
                    axis = ca.np.array(joint.axis)
                joint_frame = T.prismatic(joint.origin.xyz,
                                          joint.origin.rpy,
                                          joint.axis, q[self._joint_map[joint.name]])
                T_fk = ca.mtimes(T_fk, joint_frame)

            elif joint.type in ["revolute", "continuous"]:
                if joint.axis is None:
                    axis = ca.np.array([1., 0., 0.])
                else:
                    axis = ca.np.array(joint.axis)
                axis = (1./ca.np.linalg.norm(axis))*axis
                joint_frame = T.revolute(
                    joint.origin.xyz,
                    joint.origin.rpy,
                    joint.axis, q[self._joint_map[joint.name]])
                T_fk = ca.mtimes(T_fk, joint_frame)

        T_fk = ca.mtimes(T_fk, link_transformation)

        return {
            "T_fk": T_fk
        }
