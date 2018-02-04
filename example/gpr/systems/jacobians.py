from numpy import dot, eye, outer, tensordot, zeros
from numpy.linalg import norm

from example.gpr.variables.derivatives import dEdρ, dEdp, dEdA, dEdJ
from example.gpr.variables.derivatives import dTdρ, dTdp, dσdρ, dσdA
from example.gpr.variables.state import pressure, temperature, sigma
from example.parameters import α
from options import nV


def dPdQ(Q):
    """ Returns the Jacobian of the primitive variables with respect to the
        conserved variables
    """
    ret = eye(nV)

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3, 3])
    J = Q[14:17] / ρ
    p = pressure(ρ, E, v, A, J)

    ψ = dEdA(A)
    H = dEdJ(J)

    Γ_ = 1 / (ρ * dEdp(ρ))
    Υ = Γ_ * (norm(v)**2 + α**2 * norm(J)**2 - (E + ρ * dEdρ(ρ, p)))

    ret[1, 0] = Υ
    ret[1, 1] = Γ_
    ret[1, 2:5] = -Γ_ * v
    ret[2:5, 0] = -v / ρ

    for i in range(2, 5):
        ret[i, i] = 1 / ρ

    ret[1, 5:14] = -Γ_ * ρ * ψ.ravel()

    ret[1, 14:17] = -Γ_ * H
    ret[14:17, 0] = -J / ρ
    for i in range(14, 17):
        ret[i, i] = 1 / ρ

    return ret


def dFdP(Q, d):
    """ Returns the Jacobian of the flux vector in the dth direction with
        respect to the primitive variables
    """
    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3, 3])
    J = Q[14:17] / ρ
    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)

    σ = sigma(ρ, A)
    Eρ = dEdρ(ρ, p)
    Ep = dEdp(ρ)
    ψ = dEdA(A)
    H = dEdJ(J)
    σρ = dσdρ(ρ, A)
    σA = dσdA(ρ, A)
    Tρ = dTdρ(ρ, p)
    Tp = dTdp(ρ)

    ρvd = ρ * v[d]
    vv = outer(v, v)
    Ψ = ρ * vv - σ
    Φ = vv - σρ
    Δ = (E + ρ * Eρ) * v - dot(σρ, v) + Tρ * H
    Π = (ρ * Ep + 1) * v + Tp * H
    Ω = ρ * outer(v, ψ).reshape([3, 3, 3]) - tensordot(v, σA, axes=(0, 0))

    ret = zeros([nV, nV])

    ret[0, 0] = v[d]
    ret[0, 2 + d] = ρ

    ret[1, 0] = Δ[d]
    ret[1, 1] = Π[d]
    ret[1, 2:5] = Ψ[d]
    ret[1, 2 + d] += ρ * E + p
    ret[1, 5:14] = Ω[d].ravel()
    ret[1, 14:17] = ρvd * H
    ret[1, 14 + d] += α**2 * T

    ret[2:5, 0] = Φ[d]
    for i in range(2, 5):
        ret[i, i] = ρvd
    ret[2:5, 2 + d] += ρ * v
    ret[2 + d, 1] = 1
    ret[2:5, 5:14] = -σA[d].reshape([3, 9])

    ret[5 + d, 2:5] = A[0]
    ret[8 + d, 2:5] = A[1]
    ret[11 + d, 2:5] = A[2]
    ret[5 + d, 5:8] = v
    ret[8 + d, 8:11] = v
    ret[11 + d, 11:14] = v

    ret[14:17, 0] = v[d] * J
    ret[14 + d, 0] += Tρ
    ret[14 + d, 1] = Tp
    ret[14:17, 2 + d] = ρ * J
    for i in range(14, 17):
        ret[i, i] = ρvd

    return ret
