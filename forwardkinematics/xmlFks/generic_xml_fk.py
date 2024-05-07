from typing import Union, List, Optional
import logging
import xml.etree.ElementTree as ET
import numpy as np
import casadi as ca

from forwardkinematics.fksCommon.fk import ForwardKinematics

casadiOrNumpy = Union[ca.SX, np.ndarray]

def matrix_multiply(A: casadiOrNumpy, B: casadiOrNumpy) -> casadiOrNumpy:
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.dot(A, B)
    elif isinstance(A, ca.SX) and isinstance(B, ca.SX):
        return ca.mtimes(A, B)
    elif isinstance(A, ca.SX) and isinstance(B, np.ndarray):
        return ca.mtimes(A, B)
    elif isinstance(A, np.ndarray) and isinstance(B, ca.SX):
        return ca.mtimes(A, B)
    else:
        raise ValueError(f"A({type(A)}) and B({type(B)}) must be either casadi or numpy arrays.")

def get_T_element(
        element: ET.Element,
    ) -> np.ndarray:
    """
    Get the transformation matrix of an element.
    """
    pos = element.get('pos') if 'pos' in element.attrib else '0 0 0'
    offset = np.array([float(i) for i in pos.split()])
    if 'quat' in element.attrib:
        quat = [float(i) for i in element.attrib['quat'].split()]
        rotation_matrix = quat2rotMat(quat)
    else:
        rotation_matrix = np.eye(3)
    return homogeneous_transformation(
        rotation=rotation_matrix,
        translation=offset
    )

def quat2rotMat(quat: List[float]) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.
    """
    # Normalize the quaternion
    quat_np = np.array(quat)
    q = quat_np / np.linalg.norm(quat_np)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2]
    ])
    return R

def axis2rotMat(axis: List[int], theta: casadiOrNumpy) -> casadiOrNumpy:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if isinstance(theta, np.float64) or isinstance(theta, float):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ])
        return R
    elif isinstance(theta, ca.SX):
        axis_ca = ca.SX(axis)
        axis_ca = axis_ca / ca.sqrt(ca.dot(axis_ca, axis_ca))
        a = ca.cos(theta / 2.0)
        b, c, d = ca.vertsplit(-axis_ca * ca.sin(theta / 2.0))
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = ca.vertcat(ca.horzcat(aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)),
                       ca.horzcat(2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)),
                       ca.horzcat(2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc))
        return ca.SX(R)
    else:
        raise ValueError(f"theta {type(theta)} must be either casadi or numpy array.")

def homogeneous_transformation(
        rotation: Optional[casadiOrNumpy] = None,
        translation: Optional[casadiOrNumpy] = None,
    ) -> casadiOrNumpy:
    """
    Construct a homogeneous transformation matrix from rotation matrix and translation vector.
    """
    if rotation is not None and isinstance(rotation, ca.SX) or translation is not None and isinstance(translation, ca.SX):
        T = ca.SX.eye(4)
    else:
        T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation
    return T


class GenericXMLFk(ForwardKinematics):
    """
    Forward kinematics class for Mujoco XML files.
    """
    def __init__(self, xml: str, root_link: str, end_links: List[str], base_type: str = 'holonomic'):
        """
        Initialize the forward kinematics class.

        Parameters:
            xml (str): the XML string.
        """
        super().__init__()
        self._xml = xml
        self.tree = ET.ElementTree(ET.fromstring(xml))
        self.root = self.tree.getroot()


    def casadi(
        self, q: ca.SX,
        child_link: str,
        parent_link: Union[str, None] = None,
        link_transformation: np.ndarray = np.eye(4),
        position_only: bool = False
    ) -> ca.SX:
        return self.fk(
            q, child_link, parent_link, link_transformation, position_only
        )

    def numpy(
        self, q: np.ndarray,
        child_link: str,
        parent_link: Union[str, None] = None,
        link_transformation: np.ndarray = np.eye(4),
        position_only: bool = False
    ) -> ca.SX:
        return self.fk(
            q, child_link, parent_link, link_transformation, position_only
        )

    def fk(
        self, q: Union[ca.SX, np.ndarray],
        child_link: str,
        parent_link: Union[str, None] = None,
        link_transformation: np.ndarray = np.eye(4),
        position_only: bool = False
    ) -> casadiOrNumpy:
        """
        Compute the forward kinematics of the robot.
        
        Parameters:
            q (ca.SX): The joint configuration.
            child_link (str): The name of the child link.
            parent_link (str): The name of the parent link.
            link_transformation (np.ndarray): The transformation matrix of the parent link.
            position_only (bool): If True, only the position of the end-effector is returned.
        
        """
        root = self.root.find('.//worldbody')
        T = homogeneous_transformation()
        joint_counter = 0
        child_link_found = False
        for element in root.iter():
            if child_link_found:
                break
            T_element = np.eye(4)
            if element.tag == 'body':
                T_element = get_T_element(element)
                if element.get('name') == child_link:
                    child_link_found = True
            if element.tag == 'joint':
                joint_type = 'hinge' if 'type' not in element.attrib else element.attrib['type']
                if 'axis' not in element.attrib:
                    joint_axis = [0, 0, 1]
                    logging.warning(f"Joint axis not found in XML. Assuming default axis {joint_axis}.")
                else:
                    joint_axis = [int(x) for x in element.attrib['axis'].split()]
                if joint_type == 'hinge':
                    T_element = homogeneous_transformation(
                        rotation=axis2rotMat(
                            joint_axis,
                            q[joint_counter],
                        )
                    )
                elif joint_type == 'slide':
                    T_element = homogeneous_transformation(
                        translation=joint_axis * q[joint_counter]
                    )
                joint_counter += 1
            if element.tag == 'geom':
                if element.get('name') == child_link:
                    T_element = get_T_element(element)
                    child_link_found = True
                if element.get('mesh') == child_link:
                    T_element = get_T_element(element)
                    child_link_found = True
            T = matrix_multiply(T, T_element)
        if not child_link_found:
            raise ValueError(f"Child link {child_link} not found in XML.")
        else:
            if position_only:
                return T[0:3,3]
            else:
                return T



