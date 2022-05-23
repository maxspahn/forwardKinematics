import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk


def main():
    with open("albert.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link3')
    n = fk.n()
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = fk.fk(
        q_ca, "panda_link7", "panda_ee", positionOnly=True
    )
    fk_panda_link_np = fk.fk(
        q_np, "panda_link0", "panda_ee", positionOnly=True
    )
    print(f"casadi fk: {fk_panda_link}")
    print(f"numpy fk: {fk_panda_link_np}")


if __name__ == "__main__":
    main()
