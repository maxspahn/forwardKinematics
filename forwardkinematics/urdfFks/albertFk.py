import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class AlbertFk(URDFForwardKinematics):
    def __init__(self, n):
        fileName = "albert.urdf"
        relevantLinks = ["world", "base_tip_link", "top_mount"] + [
            "panda_link" + str(i) for i in [0, 3, 4, 5, 6, 7, 8, 9]
        ]
        super().__init__(fileName, relevantLinks, "world", len(relevantLinks) - 1)
