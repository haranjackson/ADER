from numpy import dot, exp, eye, log

from models.nonlinear_elasticity.eos import U_1, W
from models.nonlinear_elasticity.matrices import gram, I_1, I_2, I_3, inv3


def GdU_1dG(I3, K0, α):
    """ Returns G * dU1/dG
    """
    return K0 / (2 * α) * (I3**α - I3**(α / 2)) * eye(3)


def GdU_2dG(I3, S, cv, T0, γ):
    """ Returns G * dU2/dG
    """
    return cv * T0 * γ / 2 * (exp(S / cv) - 1) * I3**(γ / 2) * eye(3)


def GdWdG(G, I1, I2, I3, B0, β):
    """ Returns G * dW/dG
    """
    const = B0 / 2 * I3**(β / 2)
    return const * ((β / 2) * (I1**2 / 3 - I2) * eye(3) - I1 / 3 * G + dot(G, G))


def sigma(ρ, F, S, MP):
    """ Returns the total stress tensor
    """
    K0 = MP['K0']
    α = MP['α']
    cv = MP['cv']
    T0 = MP['T0']
    γ = MP['γ']
    B0 = MP['B0']
    β = MP['β']

    A = inv3(F)
    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)

    GdedG = GdU_1dG(I3, K0, α) + GdU_2dG(I3, S, cv, T0, γ) + GdWdG(G, I1, I2, I3, B0, β)
    return -2 * ρ * GdedG


def entropy(E, F, v, MP):

    K0 = MP['K0']
    α = MP['α']
    B0 = MP['B0']
    β = MP['β']
    cv = MP['cv']
    T0 = MP['T0']
    γ = MP['γ']

    A = inv3(F)
    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)

    U2 = E - (U_1(I3, K0, α) + W(I1, I2, I3, B0, β) + dot(v, v) / 2)
    return cv * log(1 + U2 / (cv * T0 * I3**(γ / 2)))


def pressure(ρ, A, S, MP):

    G = gram(A)
    I1 = I_1(G)
    I2 = I_2(G)
    I3 = I_3(G)

    K0 = MP['K0']
    α = MP['α']
    cv = MP['cv']
    T0 = MP['T0']
    γ = MP['γ']
    B0 = MP['B0']
    β = MP['β']
    const = B0 / 2 * I3**(β / 2)

    ret = K0 / (2 * α) * (I3**α - I3**(α / 2))
    ret += cv * T0 * γ / 2 * (exp(S / cv) - 1) * I3**(γ / 2)
    ret += const * ((β / 2) * (I1**2 / 3 - I2))
    return 2 * ρ * ret
