import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.tiagoFk import TiagoFk


def main():
    n = 20
    q_ca = ca.SX.sym("q", n)
    fkPlanar = TiagoFk()
    q_np = np.random.random(n) * 0.0
    fkCasadi = fkPlanar.fk(q_ca, 5, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 5, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)

if __name__ == "__main__":
    main()
