from typing import Union, List
import xml.etree.ElementTree as ET
import numpy as np
import casadi as ca

from forwardkinematics.fksCommon.fk import ForwardKinematics

def rotation_matrix(axis: List[int], theta: ca.SX) -> ca.SX:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis_ca = ca.SX(axis)
    axis_ca = axis_ca / ca.sqrt(ca.dot(axis, axis))
    a = ca.cos(theta / 2.0)
    b, c, d = ca.vertsplit(-axis_ca * ca.sin(theta / 2.0))
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = ca.vertcat(ca.horzcat(aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
                   ca.horzcat(2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
                   ca.horzcat(2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc))
    return ca.SX(R)

def homogeneous_transformation(rotation_matrix: ca.SX, translation: ca.SX) -> ca.SX:
    """
    Construct a homogeneous transformation matrix from rotation matrix and translation vector.
    """
    T = ca.SX.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


class XMLForwardKinematics(ForwardKinematics):
    """
    Forward kinematics class for Mujoco XML files.
    """
    def __init__(self, xml: str):
        """
        Initialize the forward kinematics class.

        Parameters:
            xml (str): the XML string.
        """
        super().__init__()
        self._xml = xml
        self.tree = ET.ElementTree(ET.fromstring(xml))
        self.root = self.tree.getroot()
        self._q_ca = ca.SX.sym("q", self._n)
        self._mount_transformation = np.identity(4)
        if base_type in ['diffdrive']:
            self._q_base = ca.SX.sym("q_base", 3)
        self.generate_functions()

    def casadi(
        self, q: ca.SX,
        child_link: str,
        parent_link: Union[str, None] = None,
        link_transformation: np.ndarray = np.eye(4),
        position_only: bool = False
    ) -> ca.SX:
        """
        Compute the forward kinematics of the robot.
        
        Parameters:
            q (ca.SX): The joint configuration.
            child_link (str): The name of the child link.
            parent_link (str): The name of the parent link.
            link_transformation (np.ndarray): The transformation matrix of the parent link.
            position_only (bool): If True, only the position of the end-effector is returned.
        
        Returns:
            ca.SX: The homogeneous transformation matrix of the end-effector.
        """
        root = self.root.find('.//worldbody')
        T = ca.SX.eye(4)
        joint_counter = 0
        child_link_found = False
        for element in root.iter():
            if child_link_found:
                return T
            T_element = ca.SX.eye(4)
            if element.tag == 'body':
                pos = element.get('pos') if 'pos' in element.attrib else '0 0 0'
                offset = ca.SX([float(i) for i in pos.split()])
                T_element = homogeneous_transformation(ca.SX.eye(3), offset)
                if element.get('name') == child_link:
                    child_link_found = True
            if element.tag == 'joint':
                joint_type = 'hinge' if 'type' not in element.attrib else element.attrib['type']
                joint_axis = [0, 0, 1] if 'axis' not in element.attrib else [int(x) for x in element.attrib['axis'].split()]
                if joint_type == 'hinge':
                    T_element = homogeneous_transformation(
                        rotation_matrix(joint_axis, q[joint_counter]),
                        ca.SX.zeros(3)
                    )
                elif joint_type == 'slide':
                    T_element = homogeneous_transformation(
                        ca.SX.eye(3),
                        joint_axis * q[joint_counter]
                    )
                joint_counter += 1
            if element.tag == 'geom':
                if element.get('name') == child_link:
                    pos = element.get('pos') if 'pos' in element.attrib else '0 0 0'
                    offset = ca.SX([float(i) for i in pos.split()])
                    T_element = homogeneous_transformation(ca.SX.eye(3), offset)
                    child_link_found = True
            T = ca.mtimes(T, T_element)
        if not child_link_found:
            raise ValueError(f"Child link {child_link} not found in XML.")
        else:
            return T



