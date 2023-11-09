import os
import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE=absolute_path + "/assets/albert.urdf"


def main():
    with open(URDF_FILE, "r") as file:
        urdf = file.read()
    fk = GenericURDFFk(urdf, root_link='panda_link3', end_links='panda_leftfinger')
    T_0 = np.identity(4)
    T_0[0:3, 3] = np.array([0.0, 0.0, 0.0])
    fk.set_mount_transformation(T_0)
    n = fk.n()
    q_ca = ca.SX.sym('q', n)
    q_np = np.zeros(n)
    fk_panda_link = fk.casadi(
        q_ca, "panda_link3", "panda_leftfinger", position_only=True
    )
    fk_panda_link_np = fk.numpy(
        q_np, "panda_link3", "panda_leftfinger", position_only=True
    )
    return fk_panda_link, fk_panda_link_np


if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_numpy)
    print(fk_casadi)
