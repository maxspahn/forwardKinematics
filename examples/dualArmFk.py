import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.dual_arm_fk import DualArmFk


def main():
    n = 5
    q_ca = ca.SX.sym("q", n)
    fkPlanar = DualArmFk()
    q_np = np.random.random(n) * 0.0
    fkCasadi = fkPlanar.fk(q_ca, 3, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 3, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)
    fkCasadi = fkPlanar.fk(q_ca, 5, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 5, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)

if __name__ == "__main__":
    main()
