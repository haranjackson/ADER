from numpy import dot, exp

from models.nonlinear_elasticity.matrices import gram, I_1, I_2, I_3


def U_1(I3, K0, α):
    """ Returns the first component of the thermal energy density,
        given the third invariant of G
    """
    return K0 / (2 * α**2) * (I3**(α / 2) - 1)**2


def U_2(I3, S, cv, T0, γ):
    """ Returns the second component of the thermal energy density,
        given the third invariant of G and entropy
    """
    return cv * T0 * I3**(γ / 2) * (exp(S / cv) - 1)


def W(I1, I2, I3, B0, β):
    """ Returns the internal energy due to shear deformations,
        given the invariants of G
    """
    return B0 / 2 * I3**(β / 2) * (I1**2 / 3 - I2)


def total_energy(A, S, v, MP):
    """ Returns the total energy, given the distortion tensor and entropy
    """
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

    e = U_1(I3, K0, α) + U_2(I3, S, cv, T0, γ) + W(I1, I2, I3, B0, β)
    return e + dot(v, v) / 2
