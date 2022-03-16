import casadi as ca
import numpy as np
from forwardkinematics.fksCommon.fk_creator import FkCreator

robotTypes = ['panda', 'boxer', 'albert', 'tiago', 'planarArm', 'mobilePanda', 'pointRobot', 'groundRobot', 'pointRobotUrdf', 'dualArm']

genericRobotTypes = ["planarArm"]

def test_fkCreator():
    for robotType in robotTypes:
        print(robotType)
        if robotType in genericRobotTypes:
            fkCreater = FkCreator(robotType, n=5)
        else:
            fkCreator = FkCreator(robotType)
        fk = fkCreator.fk()
        q_np = np.random.random(fk.n())
        fkNumpy = fk.fk(q_np, fk.n(), positionOnly=False)
        assert isinstance(fkNumpy, np.ndarray)
