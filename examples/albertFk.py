import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.albertFk import AlbertFk


def main():
    n = 10
    q_ca = ca.SX.sym("q", n)
    fkPlanar = AlbertFk()
    q_np = np.random.random(n) * 0.0
    fkCasadi = fkPlanar.fk(q_ca, 1, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 1, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)
    fk_casadi_by_name = fkPlanar.casadi_by_name(q_ca, 'panda_rightfinger', positionOnly=True)
    print(fk_casadi_by_name)
    fk_numpy_by_name = fkPlanar.numpy_by_name(q_np, 'extrusion1', positionOnly=True)
    print(fk_numpy_by_name)

if __name__ == "__main__":
    main()
