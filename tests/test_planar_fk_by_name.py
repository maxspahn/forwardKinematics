import numpy as np
from forwardkinematics.planarFks.pointFk import PointFk

def test_point_fk():
    point_fk = PointFk()
    fk_numpy_by_name = point_fk.fk(np.array([1.0, -2.1]), '01')
    fk_numpy = point_fk.fk(np.array([1.0, -2.1]), 1)
    assert fk_numpy.size == 2
    assert fk_numpy[0] == fk_numpy_by_name[0]
    assert fk_numpy[1] == fk_numpy_by_name[1]
