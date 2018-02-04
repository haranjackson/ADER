from numpy import dot

from example.gpr.variables.functions import dev, gram
from example.parameters import γ, cs, α, cv


def dEdρ(ρ, p):
    """ Returns the partial derivative of E by ρ
    """
    return - p / ((γ - 1) * ρ**2)


def dEdp(ρ):
    """ Returns the partial derivative of E by p
    """
    return 1 / ((γ - 1) * ρ)


def dEdA(A):
    """ Returns the partial derivative of E by A
    """
    G = gram(A)
    return cs**2 * dot(A, dev(G))


def dEdJ(J):
    """ Returns the partial derivative of E by J
    """
    return α**2 * J


def dTdρ(ρ, p):
    """ Returns the partial derivative of T by ρ
    """
    return dEdρ(ρ, p) / cv


def dTdp(ρ):
    """ Returns the partial derivative of T by p
    """
    return dEdp(ρ) / cv


def dσdρ(ρ, A):
    """ Returns the partial derivative of σ by ρ
    """
    return -ρ * dot(A.T, dEdA(A))


def dσdA(ρ, A):
    """ Returns the partial derivative of σ by A
    """
    G = gram(A)
    AdevGT = dot(A, dev(G)).T
    GA = dot(G[:, :, None], A[:, None])
    ret = GA.swapaxes(0, 3) + GA.swapaxes(1, 3) - 2 / 3 * GA

    for i in range(3):
        ret[i, :, :, i] += AdevGT
        ret[:, i, :, i] += AdevGT

    return -ρ * cs**2 * ret
