import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk


def main():
    with open("albert.urdf", "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, rootLink='panda_link3', end_link='panda_leftfinger')
    T_0 = np.identity(4)
    T_0[0:3, 3] = np.array([0.0, 0.0, 0.0])
    fk.set_mount_transformation(T_0)
    n = fk.n()
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = fk.fk(
        q_ca, "panda_link3", "panda_leftfinger", position_only=True
    )
    fk_panda_link_np = fk.fk(
        q_np, "panda_link3", "panda_leftfinger", position_only=True
    )
    print(f"casadi fk: {fk_panda_link}")
    print(f"numpy fk: {fk_panda_link_np}")


if __name__ == "__main__":
    main()
