import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk


def main():
    q_ca = ca.SX.sym("q", 7)
    with open("panda.urdf", "r") as file:
        urdf = file.read()
    fk_panda = GenericURDFFk(
        urdf,
        rootLink = 'panda_link0',
        end_link="panda_leftfinger",
    )
    q_np = np.random.random(7)
    fk_casadi = fk_panda.casadi_by_name(q_ca, 'panda_rightfinger', position_only=False)
    fk_numpy = fk_panda.numpy_by_name(q_np, 'panda_leftfinger', position_only=False)
    print(fk_numpy)
    print(fk_casadi)


if __name__ == "__main__":
    main()
