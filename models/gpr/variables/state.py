from numpy import dot

from models.gpr.misc.functions import dev, gram
from models.gpr.variables import mg
from models.gpr.variables.derivatives import dEdA, dEdJ
from models.gpr.variables.eos import E_2A, E_2J, E_3


def pressure(ρ, E, v, A, J, MP):

    E1 = E - E_3(v)
    E1 -= E_2A(ρ, A, MP)

    if MP.THERMAL:
        E1 -= E_2J(J, MP)

    p = mg.pressure(ρ, E1, MP)

    return p


def temperature(ρ, p, MP):

    return mg.temperature(ρ, p, MP)


def heat_flux(T, J, MP):

    H = dEdJ(J, MP)
    return H * T


def sigma(ρ, A, MP):
    """ Returns the symmetric viscous shear stress tensor
    """
    ψ = dEdA(ρ, A, MP)
    return -ρ * dot(A.T, ψ)


def dsigmadA(ρ, A, MP):
    """ Returns T_ijmn = dσ_ij / dA_mn, holding ρ constant.
        NOTE: Only valid for EOS with E_2A = cs^2/4 * |devG|^2
    """
    cs2 = MP.cs2
    G = gram(A)

    GA = dot(G[:, :, None], A[:, None])
    ret = GA.swapaxes(0, 3) + GA.swapaxes(1, 3) - 2 / 3 * GA

    AdevGT = dot(A, dev(G)).T
    for i in range(3):
        ret[i, :, :, i] += AdevGT
        ret[:, i, :, i] += AdevGT

    return -ρ * cs2 * ret


def dsigmadρ(ρ, A, MP):
    """ Returns the symmetric viscous shear stress tensor
    """
    ψ = dEdA(ρ, A, MP)
    return -dot(A.T, ψ)
