from numpy import dot

from example.gpr.variables.functions import dev, gram
from example.gpr.variables.derivatives import dEdA, dEdJ
from example.gpr.variables.eos import E_1, E_2A, E_2J, E_3
from example.parameters import γ, cs, cv


def pressure(ρ, E, v, A, J):

    E1 = E - E_2A(ρ, A) - E_2J(J) - E_3(v)
    return E1 * (γ - 1) * ρ


def temperature(ρ, p):

    return E_1(ρ, p) / cv


def heat_flux(T, J):

    H = dEdJ(J)
    return H * T


def sigma(ρ, A):
    """ Returns the symmetric viscous shear stress tensor
    """
    return -ρ * dot(A.T, dEdA(A))
