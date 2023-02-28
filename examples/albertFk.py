import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.albertFk import AlbertFk
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk


def main():
    q_ca = ca.SX.sym("q", 10)
    q_ca_2 = ca.SX.sym("q", 11)
    with open("albert.urdf", "r") as file:
        urdf = file.read()
    fkPlanar = AlbertFk()
    fk_generic = GenericURDFFk(
        urdf,
        rootLink = 'base_link',
        end_link="panda_ee",
        base_type='diffdrive',
    )
    q_np = np.random.random(10) * 0.0
    q_np_2 = np.random.random(11) * 0.0
    fkCasadi = fkPlanar.fk(q_ca, 1, positionOnly=True)
    fkNumpy = fkPlanar.fk(q_np, 1, positionOnly=True)
    print(fkNumpy)
    print(fkCasadi)
    fk_casadi_by_name = fkPlanar.casadi_by_name(q_ca, 'panda_rightfinger', positionOnly=True)
    print(fk_casadi_by_name)
    fk_numpy_by_name = fkPlanar.numpy_by_name(q_np, 'chassis_link', positionOnly=True)
    print(fk_numpy_by_name)

    fk_casadi_generic = fk_generic.numpy_by_name(q_np_2, 'chassis_link', positionOnly=True)
    print(fk_casadi_generic)

if __name__ == "__main__":
    main()
