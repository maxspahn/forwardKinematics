import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class TiagoFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "tiago.urdf"
        linkIndices = [1, 2, 3, 4, 5, 7]
        relevantLinks = ["origin", "base_link", "torso_fixed_link", "torso_fixed_link", "torso_lift_link"]
        relevantLinks += ["arm_left_" + str(i) + "_link" for i in linkIndices] + [
            "gripper_left_grasping_frame"
        ]
        relevantLinks += ["arm_right_" + str(i) + "_link" for i in linkIndices] + [
            "gripper_right_grasping_frame"
        ]
        relevantLinks += ["head_1_link", "head_2_link"]
        super().__init__(fileName, relevantLinks, "world", ["gripper_left_grasping_frame", "gripper_right_grasping_frame", "head_2_link"], len(relevantLinks) - 1)
