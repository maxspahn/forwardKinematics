import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class DualArmFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "dual_arm.urdf"
        relevantLinks = ["world", "link1", "link2", "ee1_link", "link4", "ee2_link"]
        super().__init__(fileName, relevantLinks, "world", ["ee1_link", "ee2_link"], 5)
