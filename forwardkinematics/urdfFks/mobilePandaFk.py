import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class MobilePandaFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "mobilePanda.urdf"
        relevantLinks = ["world", "base_link_y", "base_link"] + [
            "panda_link" + str(i) for i in [0, 3, 4, 5, 6, 7, 8, 9]
        ]
        super().__init__(fileName, relevantLinks, "world", "panda_link9", 10)
