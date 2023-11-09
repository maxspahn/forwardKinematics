import os
import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE=absolute_path + "/assets/panda.urdf"

def main():
    with open(URDF_FILE, "r") as file:
        urdf = file.read()
    fk_panda = GenericURDFFk(
        urdf,
        root_link = 'panda_link0',
        end_links="panda_leftfinger",
    )
    dof = fk_panda.n()
    q_np = np.random.random(dof)
    q_ca = ca.SX.sym("q", dof)
    fk_casadi = fk_panda.casadi(q_ca, 'panda_rightfinger', position_only=False)
    fk_numpy = fk_panda.numpy(q_np, 'panda_leftfinger', position_only=False)
    return fk_casadi, fk_numpy


if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_numpy)
    print(fk_casadi)
