import numpy as np
from forwardkinematics.fksCommon.fk_creator import FkCreator

fk = FkCreator('pointRobotUrdf', 3).fk()
fk_random = fk.fk(np.random.random(3), 2)
print(f"fk_random : \n{fk_random}")
