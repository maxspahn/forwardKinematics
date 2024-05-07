import os
import casadi as ca
import numpy as np
from forwardkinematics import GenericXMLFk

absolute_path = os.path.dirname(os.path.abspath(__file__))
XML_FILE=absolute_path + "/assets/xarm.xml"

def main():
    with open(XML_FILE, "r") as file:
        urdf = file.read()
    fk_panda = GenericXMLFk(
        urdf,
        root_link = 'link0',
        end_links=["link7"],
    )
    dof = 13
    q_np = np.random.random(dof)
    q_ca = ca.SX.sym("q", dof)
    fk_casadi = fk_panda.casadi(q_ca, 'link2', position_only=True)
    fk_numpy = fk_panda.numpy(q_np, 'link7', position_only=True)
    return fk_casadi, fk_numpy


if __name__ == "__main__":
    fk_casadi, fk_numpy = main()
    print(fk_casadi)
    print(fk_numpy)
