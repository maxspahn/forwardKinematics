import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk


def main():
    q_ca = ca.SX.sym("q", 11)
    q_np = np.zeros(11)
    with open("albert.urdf", "r") as file:
        urdf = file.read()
    fk_generic = GenericURDFFk(
        urdf,
        rootLink = 'base_link',
        end_link="panda_ee",
        base_type='diffdrive',
    )
    fk_casadi_by_name = fk_generic.casadi_by_name(q_ca, 'panda_rightfinger', position_only=True)
    fk_numpy_by_name = fk_generic.numpy_by_name(q_np, 'panda_rightfinger', position_only=True)

    print(fk_casadi_by_name)
    print(fk_numpy_by_name)

if __name__ == "__main__":
    main()
