import casadi as ca
import numpy as np
import pytest 
import re

from forwardkinematics.urdfFks.dual_arm_fk import DualArmFk

def get_variables_from_casadi_expression(variableName: str, exp: ca.SX, dim: int) -> str:
    """This function extract a string containing all variables in the ca.SX expression.

    Parameters
    ----------

    variableName: str
        Name of the variable that are potentially in the expression.
    exp: ca.SX
        Casadi expression to be investigated.
    dim: int
        Dimension of the variable. This is important to because casadi uses
        different namings for vectors and scalars.

    Returns
    ------------

    str
        Returns a string of form [a_1, a_3, a_5]

    """
    if dim == 1: 
        matches = re.findall(r'('+ variableName + ')', str(exp))
    else:
        matches = re.findall(r'('+ variableName + '_\d)', str(exp))
    if len(matches) < 2:
        return matches[0]
    matches = sorted(set(matches))
    variable_string = "["
    for match in matches:
        variable_string += match + ", "
    variable_string = variable_string[:-2] + "]"
    return variable_string

@pytest.fixture
def fk():
    return DualArmFk()

def test_dualArmFk(fk):
    q_ca = ca.SX.sym("q", fk.n())
    q_np = np.random.random(fk.n())
    fkCasadi = fk.fk(q_ca, fk.n(), position_only=False)
    fkNumpy = fk.fk(q_np, fk.n(), position_only=False)
    assert isinstance(fkCasadi, ca.SX)
    assert isinstance(fkNumpy, np.ndarray)

def test_different_endeffectors(fk):
    q_ca = ca.SX.sym("q", fk.n())
    fkCasadi_ee1 = fk.fk(q_ca, 3, position_only=True)
    fkCasadi_ee2 = fk.fk(q_ca, 5, position_only=True)
    assert isinstance(fkCasadi_ee1, ca.SX)
    assert isinstance(fkCasadi_ee2, ca.SX)
    variables_ee1 = get_variables_from_casadi_expression("q", fkCasadi_ee1, fk.n())
    variables_ee2 = get_variables_from_casadi_expression("q", fkCasadi_ee2, fk.n())
    assert variables_ee1 != variables_ee2
    assert variables_ee1 == "[q_0, q_1, q_2]"
    assert variables_ee2 == "[q_0, q_3, q_4]"

