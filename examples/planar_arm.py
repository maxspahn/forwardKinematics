import numpy as np
import casadi as ca
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


def main():
    fk = PlanarArmFk(10)
    q_ca = ca.SX.sym("q", fk.n())
    q_np = np.random.random(fk.n())
    fk_casadi = fk.casadi(q_ca, 5, parent_link=0, position_only=True)
    fk_numpy = fk.numpy(q_np, 5, parent_link=4, position_only=True)
    return fk_casadi, fk_numpy

if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_casadi)
    print(fk_numpy)

