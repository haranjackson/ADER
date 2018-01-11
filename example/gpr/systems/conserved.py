from numpy import dot

from example.gpr.variables.eos import dEdA, dEdJ
from example.gpr.variables.sources import theta1inv, theta2inv
from example.gpr.variables.state import pressure, sigma, temperature, heat_flux


def flux_cons_ref(ret, Q, d):

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3,3])
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
    ret[2+d] += p

    σd = σ[d]
    ret[1] -= dot(σd, v)
    ret[2:5] -= σd

    Av = dot(A, v)
    ret[5+d] += Av[0]
    ret[8+d] += Av[1]
    ret[11+d] += Av[2]

    ret[1] += q[d]
    ret[14:17] += ρvd * J
    ret[14+d] += T

def block_cons_ref(ret, Q, d):

    v = Q[2:5] / Q[0]
    vd = v[d]

    for i in range(5,14):
        ret[i,i] = vd
    ret[5+d, 5+d:8+d] -= v
    ret[8+d, 8+d:11+d] -= v
    ret[11+d, 11+d:14+d] -= v

def source_cons_ref(ret, Q):

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3,3])
    J = Q[14:17] / ρ

    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)

    ψ = dEdA(ρ, A)
    H = dEdJ(J)

    θ1_1 = theta1inv(ρ, A)
    ret[5:14] = - ψ.ravel() * θ1_1

    θ2_1 = theta2inv(ρ, T)
    ret[14:17] = - ρ * H * θ2_1

def B0dot(ret, x, v):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    ret[5] = - v1 * x[6] - v2 * x[7]
    ret[6] = v0 * x[6]
    ret[7] = v0 * x[7]
    ret[8] = - v1 * x[9] - v2 * x[10]
    ret[9] = v0 * x[9]
    ret[10] = v0 * x[10]
    ret[11] = - v1 * x[12] - v2 * x[13]
    ret[12] = v0 * x[12]
    ret[13] = v0 * x[13]

def B1dot(ret, x, v):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    ret[5] = v1 * x[5]
    ret[6] = - v0 * x[5] - v2 * x[7]
    ret[7] = v1 * x[7]
    ret[8] = v1 * x[8]
    ret[9] = - v0 * x[8] - v2 * x[10]
    ret[10] = v1 * x[10]
    ret[11] = v1 * x[11]
    ret[12] = - v0 * x[11] - v2 * x[13]
    ret[13] = v1 * x[13]

def B2dot(ret, x, v):
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    ret[5] = v2 * x[5]
    ret[6] = v2 * x[6]
    ret[7] = - v0 * x[5] - v1 * x[6]
    ret[8] = v2 * x[8]
    ret[9] = v2 * x[9]
    ret[10] = - v0 * x[8] - v1 * x[9]
    ret[11] = v2 * x[11]
    ret[12] = v2 * x[12]
    ret[13] = - v0 * x[11] - v1 * x[12]

def Bdot_cons(ret, x, Q, d):

    v = Q[2:5] / Q[0]
    if d==0:
        B0dot(ret, x, v)
    elif d==1:
        B1dot(ret, x, v)
    else:
        B2dot(ret, x, v)
