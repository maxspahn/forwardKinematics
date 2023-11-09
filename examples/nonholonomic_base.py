import os
import casadi as ca
import numpy as np
from forwardkinematics import GenericURDFFk

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE=absolute_path + "/assets/boxer.urdf"


def main():
    with open(URDF_FILE, "r") as file:
        urdf = file.read()
    fk_generic = GenericURDFFk(
        urdf,
        root_link = 'base_link',
        end_links="ee_link",
        base_type='diffdrive',
    )
    dof = fk_generic.n()
    q_ca = ca.SX.sym("q", dof)
    q_np = np.random.random(dof)
    fk_casadi = fk_generic.casadi(q_ca, parent_link='base_link', child_link='ee_link', position_only=True)
    fk_numpy = fk_generic.numpy(q_np, parent_link='base_link', child_link='ee_link', position_only=True)
    return fk_casadi, fk_numpy

if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_numpy)
    print(fk_casadi)
