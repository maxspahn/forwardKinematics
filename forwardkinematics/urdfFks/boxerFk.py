import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class BoxerFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "boxer.urdf"
        relevantLinks = ["origin", "ee_link", "ee_link", "ee_link"]
        super().__init__(fileName, relevantLinks, "world", "ee_link", 3)
