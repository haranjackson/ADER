from numpy import dot, sum

from models.gpr.variables import mg
from models.gpr.misc.functions import dev, gram


def E_1(ρ, p, MP):
    """ Returns the microscale energy
    """
    return mg.internal_energy(ρ, p, MP)


def E_2A(ρ, A, MP):
    """ Returns the mesoscale energy dependent on the distortion
    """
    G = gram(A)
    devG = dev(G)
    return MP.cs2 / 4 * sum(devG * devG)


def E_2J(J, MP):
    """ Returns the mesoscale energy dependent on the thermal impulse
    """
    cα2 = MP.cα2
    return cα2 / 2 * dot(J, J)


def E_3(v):
    """ Returns the macroscale kinetic energy
    """
    return dot(v, v) / 2


def total_energy(ρ, p, v, A, J, MP):
    """ Returns the total energy
    """
    ret = E_1(ρ, p, MP) + E_2A(ρ, A, MP) + E_3(v)

    if MP.THERMAL:
        ret += E_2J(J, MP)

    return ret
