from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve

from solvers.basis import NODES, WGHTS
from system import max_abs_eigs
from system import Bdot, system
from options import N1, nV


def Bint(qL, qR, d):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(nV)
    qJump = qR - qL
    for i in range(N1):
        q = qL + NODES[i] * qJump
        tmp  = zeros(nV)
        Bdot(tmp, qJump, q, d)
        ret += WGHTS[i] * tmp
    return ret

def Aint(qL, qR, d):
    """ Returns the Osher-Solomon jump matrix for A, in the dth direction
    """
    ret = zeros(nV, dtype=complex128)
    Δq = qR - qL
    for i in range(N1):
        q = qL + NODES[i] * Δq
        J = system(q, d)
        λ, R = eig(J, overwrite_a=1, check_finite=0)
        b = solve(R, Δq, check_finite=0)
        ret += WGHTS[i] * dot(R, abs(λ)*b)
    return ret.real

def Smax(qL, qR, d):
    """ Returns the Rusanov contribution to the flux, in the dth direction
    """
    max1 = max_abs_eigs(qL, d)
    max2 = max_abs_eigs(qR, d)
    return max(max1, max2) * (qR - qL)
