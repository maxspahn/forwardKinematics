import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.pandaFk import PandaFk

def main():
    q_ca = ca.SX.sym("q", 7)
    fk_panda = PandaFk()
    q_np = np.random.random(7)
    fk_casadi = fk_panda.fk(q_ca, 2, position_only=False)
    fk_numpy = fk_panda.fk(q_np, 2, position_only=False)
    print(fk_numpy)
    print(fk_casadi)
    fk_casadi_by_name = fk_panda.fk_by_name(q_ca, 'panda_link3', position_only=True)
    print(fk_casadi_by_name)

if __name__ == "__main__":
    main()
