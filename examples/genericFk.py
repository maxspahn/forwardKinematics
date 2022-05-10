import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk


def main():
    n = 10
    q_ca = ca.SX.sym("q", n)
    q_np = np.zeros(n)
    with open("albert.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf)
    fk_panda_link = fk.fk_by_name(
        q_ca, "panda_link7", "panda_ee", positionOnly=True
    )
    fk_panda_link_np = fk.fk_by_name(
        q_np, "panda_link0", "panda_ee", positionOnly=True
    )
    print(fk_panda_link)


if __name__ == "__main__":
    main()
