import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.albertFk import AlbertFk


def main():
    n = 10
    q_ca = ca.SX.sym("q", n)
    fkPlanar = AlbertFk(n)
    q_np = np.random.random(n) * 0.0
    fkCasadi = fkPlanar.fk(q_ca, 1, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 1, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)

if __name__ == "__main__":
    main()
