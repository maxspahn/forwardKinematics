import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class PandaFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "panda.urdf"
        relevantLinks = ["panda_link" + str(i) for i in [0, 3, 4, 5, 6, 7, 8, 9]]
        super().__init__(fileName, relevantLinks, "panda_link0", "panda_link9", 7)
