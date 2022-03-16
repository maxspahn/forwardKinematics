import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class TiagoFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "tiago.urdf"
        linkIndices = [1, 2, 3, 4, 5, 7]
        relevantLinks = ["world", "base_link", "torso_fixed_link", "torso_fixed_link", "torso_lift_link"]
        relevantLinks += ["arm_left_" + str(i) + "_link" for i in linkIndices] + [
            "arm_left_tool_link"
        ]
        relevantLinks += ["arm_right_" + str(i) + "_link" for i in linkIndices] + [
            "arm_right_tool_link"
        ]
        relevantLinks += ["head_1_link", "head_2_link"]
        super().__init__(fileName, relevantLinks, "world", len(relevantLinks) - 1)
