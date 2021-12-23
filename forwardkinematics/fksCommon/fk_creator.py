import casadi as ca


class FkCreator(object):
    def __init__(self, robotType, n):
        if robotType == 'panda':
            from forwardKinematics.urdfFks.pandaFk import PandaFk
            self._fk = PandaFk(n)
        elif robotType == 'boxer':
            from forwardKinematics.urdfFks.boxerFk import BoxerFk
            self._fk = BoxerFk(n)
        elif robotType == 'tiago':
            from forwardKinematics.urdfFks.tiagoFk import TiagoFk
            self._fk = TiagoFk(n-4)
        elif robotType == 'planarArm':
            from forwardKinematics.planarFks.planarArmFk import PlanarArmFk
            self._fk = PlanarArmFk(n)
        elif robotType == 'mobilePanda':
            from forwardKinematics.urdfFks.mobilePandaFk import MobilePandaFk
            self._fk = MobilePandaFk(n)
        elif robotType == 'pointRobot' or robotType == 'pointMass':
            from forwardKinematics.planarFks.pointFk import PointFk
            self._fk = PointFk(n)
        elif robotType == 'groundRobot':
            from forwardKinematics.planarFks.groundRobotFk import GroundRobotFk
            self._fk = GroundRobotFk(n)

    def fk(self):
        return self._fk


if __name__ == "__main__":
    fk = FkCreator('boxer', 3).fk()
    q_ca = ca.SX.sym('q', fk.n())
    fk = fk.fk(q_ca, fk.n(), positionOnly=True)
    print(fk)
