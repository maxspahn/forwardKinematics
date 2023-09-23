import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk


def main():
    n = 3
    q_ca = ca.SX.sym("q", n)
    with open("boxer.urdf", "r") as file:
        urdf = file.read()
    fk_generic = GenericURDFFk(
        urdf,
        rootLink = 'base_link',
        end_link="ee_link",
        base_type='diffdrive',
    )
    q_np = np.random.random(n)
    fk_casadi = fk_generic.fk(q_ca, parent_link='base_link', child_link='ee_link', positionOnly=True)
    fk_numpy = fk_generic.fk(q_np, parent_link='base_link', child_link='ee_link', positionOnly=True)
    print(fk_casadi)
    print(fk_numpy)

if __name__ == "__main__":
    main()
