from numpy import dot, sqrt, zeros
from numpy.linalg import eigvals

from models.gpr.systems.eigenvalues import Xi1, Xi2
from models.gpr.systems.jacobians import dFdP, dPdQ
from models.gpr.misc.structures import State


def F_gpr(Q, d, MP):

    ret = zeros(17)

    P = State(Q, MP)

    ρ = P.ρ
    p = P.p()
    E = P.E
    v = P.v

    vd = v[d]
    ρvd = ρ * vd

    ret[0] = ρvd
    ret[1] = ρvd * E + p * vd
    ret[2:5] = ρvd * v
    ret[2 + d] += p

    A = P.A
    σ = P.σ()

    σd = σ[d]
    ret[1] -= dot(σd, v)
    ret[2:5] -= σd

    Av = dot(A, v)
    ret[5 + d] = Av[0]
    ret[8 + d] = Av[1]
    ret[11 + d] = Av[2]

    if MP.THERMAL:

        cα2 = MP.cα2

        J = P.J
        T = P.T()
        q = P.q()

        ret[1] += q[d]
        ret[14:17] = ρvd * J
        ret[14 + d] += T

    return ret


def S_gpr(Q, MP):

    ret = zeros(17)

    P = State(Q, MP)

    ρ = P.ρ

    ψ = P.ψ()
    θ1_1 = P.θ1_1()
    ret[5:14] = -ψ.ravel() * θ1_1

    if MP.THERMAL:
        H = P.H()
        θ2_1 = P.θ2_1()
        ret[14:17] = -ρ * H * θ2_1

    return ret


def B_gpr(Q, d, MP):

    ret = zeros([17, 17])
    P = State(Q, MP)

    v = P.v
    vd = v[d]

    for i in range(5, 14):
        ret[i, i] = vd
    ret[5 + d, 5 + d:8 + d] -= v
    ret[8 + d, 8 + d:11 + d] -= v
    ret[11 + d, 11 + d:14 + d] -= v

    return ret


def M_gpr(Q, d, MP):
    """ Returns the Jacobian in the dth direction
    """
    P = State(Q, MP)
    DFDP = dFdP(P, d)
    DPDQ = dPdQ(P)
    B = B_gpr(Q, d, MP)
    return dot(DFDP, DPDQ) + B


def max_eig_gpr(Q, d, MP):
    """ Returns maximum absolute value of the eigenvalues of the GPR system
    """
    P = State(Q, MP)
    vd = P.v[d]
    Ξ1 = Xi1(P, d)
    Ξ2 = Xi2(P, d)
    O = dot(Ξ1, Ξ2)

    lam = sqrt(eigvals(O).max())

    if vd > 0:
        return vd + lam
    else:
        return lam - vd
