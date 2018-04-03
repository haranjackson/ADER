from numpy import sqrt

from models.gpr.variables.derivatives import dEdρ, dEdp


def c_0(ρ, p, A, MP):
    """ Returns the adiabatic sound speed for the Mie-Gruneisen EOS
    """
    dE_dρ = dEdρ(ρ, p, A, MP)
    dE_dp = dEdp(ρ, MP)
    return sqrt((p / ρ**2 - dE_dρ) / dE_dp)


def c_h(ρ, T, MP):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    cα2 = MP.cα2
    cv = MP.cv
    return sqrt(cα2 * T / cv) / ρ
