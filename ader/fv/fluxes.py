from warnings import catch_warnings, simplefilter

from numpy import complex128, dot, zeros
from scipy.linalg import eig, solve


def B_INT(obj, qL, qR, d):
    """ Returns the jump matrix for B, in the dth direction.
    """
    ret = zeros(obj.NV)
    Δq = qR - qL
    for i in range(obj.N):
        q = qL + obj.NODES[i] * Δq
        B = obj.B(q, d, obj.pars)
        ret += obj.WGHTS[i] * dot(B, Δq)
    return ret


def D_OSH(obj, qL, qR, d):
    """ Returns the Osher flux component, in the dth direction
    """
    ret = zeros(obj.NV, dtype=complex128)
    Δq = qR - qL

    with catch_warnings():
        simplefilter("ignore")

        for i in range(obj.N):
            q = qL + obj.NODES[i] * Δq
            M = obj.M(q, d, obj.pars)
            λ, R = eig(M, overwrite_a=1, check_finite=False)
            b = solve(R, Δq, check_finite=False)
            ret += obj.WGHTS[i] * dot(R, abs(λ) * b)

    fL = obj.F(qL, d, obj.pars)
    fR = obj.F(qR, d, obj.pars)

    return fL + fR - ret.real


def D_ROE(obj, qL, qR, d):
    """ Returns the Roe flux component, in the dth direction
    """
    M = zeros([obj.NV, obj.NV])
    Δq = qR - qL

    for i in range(obj.N):
        q = qL + obj.NODES[i] * Δq
        M += obj.WGHTS[i] * obj.M(q, d, obj.pars)

    λ, R = eig(M, overwrite_a=1, check_finite=False)
    with catch_warnings():
        simplefilter("ignore")
        b = solve(R, Δq, overwrite_b = True, check_finite=False)

    fL = obj.F(qL, d, obj.pars)
    fR = obj.F(qR, d, obj.pars)

    return fL + fR - dot(R, abs(λ) * b).real


def D_RUS(obj, qL, qR, d):
    """ Returns the Rusanov flux component, in the dth direction
    """
    max1 = obj.max_eig(qL, d, obj.pars)
    max2 = obj.max_eig(qR, d, obj.pars)

    fL = obj.F(qL, d, obj.pars)
    fR = obj.F(qR, d, obj.pars)

    return fL + fR - max(max1, max2) * (qR - qL)
