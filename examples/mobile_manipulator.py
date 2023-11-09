import os
import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE=absolute_path + "/assets/albert.urdf"


def main():
    with open(URDF_FILE, "r") as file:
        urdf = file.read()
    fk_generic = GenericURDFFk(
        urdf,
        root_link = 'base_link',
        end_links="panda_ee",
        base_type='diffdrive',
    )
    dof = fk_generic.n()
    q_ca = ca.SX.sym("q", dof)
    q_np = np.zeros(dof)
    fk_casadi = fk_generic.casadi(q_ca, 'panda_rightfinger', position_only=True)
    fk_numpy = fk_generic.numpy(q_np, 'panda_rightfinger', position_only=True)
    return fk_casadi, fk_numpy


if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_casadi)
    print(fk_numpy)
