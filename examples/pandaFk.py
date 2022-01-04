import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.pandaFk import PandaFk


def main():
    q_ca = ca.SX.sym("q", 7)
    fkPlanar = PandaFk(7)
    q_np = np.random.random(7)
    fkCasadi = fkPlanar.fk(q_ca, 2, positionOnly=False)
    fkNumpy = fkPlanar.fk(q_np, 2, positionOnly=False)
    print(fkNumpy)
    print(fkCasadi)

if __name__ == "__main__":
    main()
