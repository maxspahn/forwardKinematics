from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics


class JackalFk(URDFForwardKinematics):
    def __init__(self):
        fileName = "jackal.urdf"
        relevantLinks = ["origin", "ee_link", "ee_link", "ee_link"]
        super().__init__(fileName, relevantLinks, "world", "ee_link", 3)
