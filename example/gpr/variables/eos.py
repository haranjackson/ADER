from numpy.linalg import norm

from example.gpr.variables.functions import dev, gram
from example.parameters import γ, cs, α, cv


def E_1(ρ, p):
    """ Returns the microscale energy using the Ideal Gas EOS
    """
    return p / ((γ - 1) * ρ)


def E_2A(ρ, A):
    """ Returns the mesoscale energy dependent on the distortion
    """
    G = gram(A)
    return cs**2 / 4 * norm(dev(G))**2


def E_2J(J):
    """ Returns the mesoscale energy dependent on the thermal impulse
    """
    return α**2 / 2 * norm(J)**2


def E_3(v):
    """ Returns the macroscale kinetic energy
    """
    return norm(v)**2 / 2


def total_energy(ρ, p, v, A, J):
    """ Returns the total energy
    """
    return E_1(ρ, p) + E_2A(ρ, A) + E_2J(J) + E_3(v)
