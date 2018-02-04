from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS
from system import max_abs_eigs
from system import block, system
from options import N, nV


def B_INT(qL, qR, d):
    """ Returns the jump matrix for B, in the dth direction.
    """
    B = zeros([nV, nV])
    tmp = zeros([nV, nV])
    Δq = qR - qL
    for i in range(N):
        q = qL + NODES[i] * Δq
        block(tmp, q, d)
        B += WGHTS[i] * tmp
    return dot(B, Δq)


def D_OSH(qL, qR, d):
    """ Returns the Osher flux component, in the dth direction
    """
    ret = zeros(nV, dtype=complex128)
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M = system(q, d)
        λ, R = eig(M, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += WGHTS[i] * dot(R, abs(λ) * b)

    return ret.real


def D_ROE(qL, qR, d):
    """ Returns the Roe flux component, in the dth direction
    """
    M = zeros([nV, nV])
    Δq = qR - qL

    for i in range(N):
        q = qL + NODES[i] * Δq
        M += WGHTS[i] * system(q, d)

    λ, R = eig(M, overwrite_a=1, check_finite=0)
    b = solve(R, Δq, check_finite=0)
    return dot(R, abs(λ) * b).real


def D_RUS(qL, qR, d):
    """ Returns the Rusanov flux component, in the dth direction
    """
    max1 = max_abs_eigs(qL, d)
    max2 = max_abs_eigs(qR, d)
    return max(max1, max2) * (qR - qL)
