import casadi as ca


class FkCreator(object):
    def __init__(self, robotType, n=3):
        if robotType == 'panda':
            from forwardkinematics.urdfFks.pandaFk import PandaFk
            self._fk = PandaFk()
        elif robotType == 'boxer':
            from forwardkinematics.urdfFks.boxerFk import BoxerFk
            self._fk = BoxerFk()
        elif robotType == 'jackal':
            from forwardkinematics.urdfFks.jackalFk import JackalFk
            self._fk = JackalFk()
        elif robotType == 'albert':
            from forwardkinematics.urdfFks.albertFk import AlbertFk
            self._fk = AlbertFk()
        elif robotType == 'tiago':
            from forwardkinematics.urdfFks.tiagoFk import TiagoFk
            self._fk = TiagoFk()
        elif robotType == 'planarArm':
            from forwardkinematics.planarFks.planarArmFk import PlanarArmFk
            self._fk = PlanarArmFk(n)
        elif robotType == 'mobilePanda':
            from forwardkinematics.urdfFks.mobilePandaFk import MobilePandaFk
            self._fk = MobilePandaFk()
        elif robotType == 'pointRobot' or robotType == 'pointMass':
            from forwardkinematics.planarFks.pointFk import PointFk
            self._fk = PointFk()
        elif robotType == 'groundRobot':
            from forwardkinematics.planarFks.groundRobotFk import GroundRobotFk
            self._fk = GroundRobotFk()
        elif robotType == 'pointRobotUrdf':
            from forwardkinematics.urdfFks.pointRobotUrdfFk import PointRobotUrdfFk
            self._fk = PointRobotUrdfFk()
        elif robotType == 'dualArm':
            from forwardkinematics.urdfFks.dual_arm_fk import DualArmFk
            self._fk = DualArmFk()

    def fk(self):
        return self._fk
