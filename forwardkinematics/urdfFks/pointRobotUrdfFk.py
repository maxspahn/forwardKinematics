import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class PointRobotUrdfFk(URDFForwardKinematics):
    def __init__(self, n):
        fileName = "pointRobot.urdf"
        relevantLinks = ["world", "base_link", "base_link", "base_link"]
        super().__init__(fileName, relevantLinks, "world", 3)
