import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class PointRobotUrdfFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "pointRobot.urdf"
        relevantLinks = ["origin", "base_link", "base_link", "base_link"]
        super().__init__(fileName, relevantLinks, "world", "base_link", 3)
