from numpy import sqrt

from example.parameters import γ, α, cv


def c_0(ρ, p):
    """ Returns the adiabatic sound speed for the ideal gas EOS
    """
    return sqrt(γ * p / ρ)


def c_h(ρ, T):
    """ Returns the velocity of the heat characteristic at equilibrium
    """
    return sqrt(α**2 * T / cv) / ρ
