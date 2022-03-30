import casadi as ca
import numpy as np
from forwardkinematics.urdfFks.pandaFk import PandaFk


def main():
    q_ca = ca.SX.sym("q", 7)
    fkPanda = PandaFk()
    q_np = np.random.random(7)
    fkCasadi = fkPanda.fk(q_ca, 2, positionOnly=False)
    fkNumpy = fkPanda.fk(q_np, 2, positionOnly=False)
    print(fkNumpy)
    print(fkCasadi)
    fkCasadiByName = fkPanda.fk_by_name(q_ca, 'panda_link3', positionOnly=True)
    print(fkCasadiByName)

if __name__ == "__main__":
    main()
