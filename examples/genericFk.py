import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.urdfFk import URDFForwardKinematics
from forwardkinematics.urdfFks.generic_fk import GenericFk

def main():
    n = 10
    q_ca = ca.SX.sym("q", n)
    q_np = np.zeros(n)
    fk = GenericFk("albert.urdf")
    fk_panda_link = fk.fk_by_name(q_ca, "panda_link7", "panda_ee", positionOnly=True)
    fk_panda_link_np = fk.fk_by_name(q_np, "panda_link0", "panda_ee", positionOnly=True)
    print(fk_panda_link)

if __name__ == "__main__":
    main()
