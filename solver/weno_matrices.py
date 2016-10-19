from itertools import product

from numpy import polyint, zeros, floor, ceil

from options import N
from auxiliary.basis import basis_polys


_, ψDer, ψInt = basis_polys()

if N%2:
    nStencils = 4
else:
    nStencils = 3


def coefficient_matrices():
    """ Generate linear systems governing the coefficients of the basis polynomials
    """
    floorHalfN = floor(N/2)
    ceilHalfN = ceil(N/2)
    Mc = zeros([nStencils, N+1, N+1])
    for p in range(N+1):
        ψpInt = ψInt[p]
        for e in range(N+1):
            if nStencils==3:
                Mc[0,e,p] = ψpInt(e-floorHalfN+1) - ψpInt(e-floorHalfN)
                Mc[1,e,p] = ψpInt(e-N+1) - ψpInt(e-N)
                Mc[2,e,p] = ψpInt(e+1) - ψpInt(e)
            else:
                Mc[0,e,p] = ψpInt(e-floorHalfN+1) - ψpInt(e-floorHalfN)
                Mc[1,e,p] = ψpInt(e-ceilHalfN+1) - ψpInt(e-ceilHalfN)
                Mc[2,e,p] = ψpInt(e-N+1) - ψpInt(e-N)
                Mc[3,e,p] = ψpInt(e+1) - ψpInt(e)
    return Mc

def oscillation_indicator():
    """ Generate the oscillation indicator matrix
    """
    Σ = zeros([N+1, N+1])
    for a in range(1,N+1):
        ψDera = ψDer[a]
        for p, m in product(range(N+1), range(N+1)):
            antiderivative = polyint(ψDera[p] * ψDera[m])
            Σ[p,m] += antiderivative(1) - antiderivative(0)
    return Σ
