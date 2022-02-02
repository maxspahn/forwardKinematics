import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class BoxerFk(URDFForwardKinematics):
    def __init__(self, n):
        fileName = "boxer.urdf"
        relevantLinks = ["world", "ee_link", "ee_link", "ee_link"]
        super().__init__(fileName, relevantLinks, "world", 3)
