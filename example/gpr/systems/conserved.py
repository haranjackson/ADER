from numpy import dot, zeros

from example.gpr.systems.jacobians import dFdP, dPdQ
from example.gpr.variables.derivatives import dEdA, dEdJ
from example.gpr.variables.sources import theta1inv, theta2inv
from example.gpr.variables.state import pressure, sigma, temperature, heat_flux
from options import nV


def flux_cons(ret, Q, d):

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3, 3])
    J = Q[14:17] / ρ

    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)
    q = heat_flux(T, J)
    σ = sigma(ρ, A)

    vd = v[d]
    ρvd = ρ * vd

    ret[0] += ρ * vd
    ret[1] += ρvd * E + p * vd
    ret[2:5] += ρvd * v
    ret[2 + d] += p

    σd = σ[d]
    ret[1] -= dot(σd, v)
    ret[2:5] -= σd

    Av = dot(A, v)
    ret[5 + d] += Av[0]
    ret[8 + d] += Av[1]
    ret[11 + d] += Av[2]

    ret[1] += q[d]
    ret[14:17] += ρvd * J
    ret[14 + d] += T


def block_cons(ret, Q, d):

    v = Q[2:5] / Q[0]
    vd = v[d]

    for i in range(5, 14):
        ret[i, i] = vd
    ret[5 + d, 5 + d:8 + d] -= v
    ret[8 + d, 8 + d:11 + d] -= v
    ret[11 + d, 11 + d:14 + d] -= v


def source_cons(ret, Q):

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3, 3])
    J = Q[14:17] / ρ

    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)

    ψ = dEdA(A)
    H = dEdJ(J)

    θ1_1 = theta1inv(ρ, A)
    ret[5:14] = - ψ.ravel() * θ1_1

    θ2_1 = theta2inv(ρ, T)
    ret[14:17] = - ρ * H * θ2_1


def system_cons(Q, d):

    DFDP = dFdP(Q, d)
    DPDQ = dPdQ(Q)
    B = zeros([nV, nV])
    block_cons(B, Q, d)
    return dot(DFDP, DPDQ) + B
