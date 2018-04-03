from models.gpr.variables import mg
from models.gpr.misc.functions import dev, dot, gram


def dEdρ(ρ, p, A, MP):
    """ Returns the partial derivative of E by ρ (holding p,A constant)
    """
    ret = mg.dedρ(ρ, p, MP)
    return ret


def dEdp(ρ, MP):
    """ Returns the partial derivative of E by p (holding ρ constant)
    """
    return mg.dedp(ρ, MP)


def dEdA(ρ, A, MP):
    """ Returns the partial derivative of E by A (holding ρ,s constant)
    """
    G = gram(A)
    return MP.cs2 * dot(A, dev(G))


def dEdJ(J, MP):
    """ Returns the partial derivative of E by J
    """
    cα2 = MP.cα2
    return cα2 * J


def dTdρ(ρ, p, MP):
    """ Returns the partial derivative of T by ρ
    """
    return mg.dTdρ(ρ, p, MP)


def dTdp(ρ, MP):
    """ Returns the partial derivative of T by p
    """
    return mg.dTdp(ρ, MP)
