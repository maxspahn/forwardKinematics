import numpy as np
from forwardkinematics.fksCommon.fk_creator import FkCreator

robotTypes = [
    "planarArm",
    "pointRobot",
    "groundRobot",
]

genericRobotTypes = ["planarArm"]


def test_fkCreator():
    for robotType in robotTypes:
        print(robotType)
        if robotType in genericRobotTypes:
            fk_creator = FkCreator(robotType, n=5)
        else:
            fk_creator = FkCreator(robotType)
        fk = fk_creator.fk()
        q_np = np.random.random(fk.n())
        fkNumpy = fk.fk(q_np, fk.n(), position_only=False)
        assert isinstance(fkNumpy, np.ndarray)
