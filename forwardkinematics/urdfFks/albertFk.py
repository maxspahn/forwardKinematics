import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class AlbertFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "albert.urdf"
        relevantLinks = ["origin", "base_tip_link", "top_mount"] + [
            "panda_link" + str(i) for i in [0, 3, 4, 5, 6, 7, 8, 9]
        ]
        super().__init__(fileName, relevantLinks, "world", ["panda_link9", "base_tip_link"], len(relevantLinks) - 1)
