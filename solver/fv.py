from itertools import product

from numpy import array, dot, zeros, tensordot

from options import ndim, dxi, N, method, n
from system import block, source
from .fv_fluxes import Dos, Drus
from auxiliary.basis import quad, end_values, derivative_values


nodes, _, weights = quad()
endVals = end_values()
derivs = derivative_values()

if method == 'osher':
    D = Dos
elif method == 'rusanov':
    D = Drus


def endpoints(qh):
    """ Returns an array containing the values of the basis polynomials at 0 and 1
        qEnd[d,e,i,j,k,:,:,:,:,:] is the set of coefficients at end e in the dth direction
    """
    nx, ny, nz = qh.shape[:3]
    qh0 = qh.reshape([nx, ny, nz] + [N+1]*(ndim+1) + [n])
    qEnd = zeros([ndim, 2, nx, ny, nz] + [N+1]*ndim + [n])
    for d in range(ndim):
        qEnd[d] = tensordot(endVals, qh0, (0,4+d))
    return qEnd

def interface(qEndL, qEndM, qEndR, d):
    """ Returns flux term and jump term in dth direction at the interface between states qhL, qhR
    """
    ret = zeros(n)
    for a in range(N+1):
        if ndim > 1:
            for b in range(N+1):
                if ndim > 2:
                    for c in range(N+1):
                        qL1 = qEndL[d, 1, a, b, c]
                        qM0 = qEndM[d, 0, a, b, c]
                        qM1 = qEndM[d, 1, a, b, c]
                        qR0 = qEndR[d, 0, a, b, c]
                        weight = weights[a] * weights[b] * weights[c]
                        ret += weight * (D(qM1, qR0, d, 1) + D(qM0, qL1, d, 0))
                else:
                    qL1 = qEndL[d, 1, a, b]
                    qM0 = qEndM[d, 0, a, b]
                    qM1 = qEndM[d, 1, a, b]
                    qR0 = qEndR[d, 0, a, b]
                    weight = weights[a] * weights[b]
                    ret += weight * (D(qM1, qR0, d, 1) + D(qM0, qL1, d, 0))
        else:
            qL1 = qEndL[d, 1, a]
            qM0 = qEndM[d, 0, a]
            qM1 = qEndM[d, 1, a]
            qR0 = qEndR[d, 0, a]
            ret += weights[a] * (D(qM1, qR0, d, 1) + D(qM0, qL1, d, 0))

    return 0.5 * ret

def center(qhijk, t, coords):
    """ Returns the space-time averaged source term and non-conservative term in cell ijk
    """
    x, y, z = coords
    qxi = zeros([ndim, N+1, n])
    if ndim > 1:
        if ndim > 2:
            qxi[0] = qhijk[t, :, y, z]
            qxi[1] = qhijk[t, x, :, z]
            qxi[2] = qhijk[t, x, y, :]
            q = qhijk[t, x, y, z]
        else:
            qxi[0] = qhijk[t, :, y]
            qxi[1] = qhijk[t, x, :]
            q = qhijk[t, x, y]
    else:
        qxi[0] = qhijk[t, :]
        q = qhijk[t, x]

    ret = source(q)

    for d in range(ndim):
        dqdxi = dot(derivs, qxi[d])[coords[d]]
        ret -= dot(block(q, d), dqdxi) / dxi[d]

    return ret

def fv_terms(qh, dt):
    """ Returns the space-time averaged interface terms, jump terms, source terms, and
        non-conservative terms
    """
    qh0 = qh.copy()
    if ndim < 3:
        qh0 = qh0.repeat([3], axis=2)
        if ndim < 2:
            qh0 = qh0.repeat([3], axis=1)

    nx, ny, nz = array(qh0.shape[:3]) - 2

    qEnd = endpoints(qh0)
    qh0 = qh0.reshape([nx+2, ny+2, nz+2] + [N+1]*(ndim+1) + [n])

    S = zeros([nx, ny, nz, n])
    F = zeros([ndim, nx, ny, nz, n])

    for i, j, k in product(range(nx), range(ny), range(nz)):

        qhijk = qh0[i+1, j+1, k+1]
        for t in range(N+1):
            for x in range(N+1):
                if ndim > 1:
                    for y in range(N+1):
                        if ndim > 2:
                            for z in range(N+1):
                                weight = weights[t] * weights[x] * weights[y] * weights[z]
                                S[i, j, k] += weight * center(qhijk, t, [x, y, z])
                        else:
                            weight = weights[t] * weights[x] * weights[y]
                            S[i, j, k] += weight * center(qhijk, t, [x, y, 0])
                else:
                    weight = weights[t] * weights[x]
                    S[i, j, k] += weight * center(qhijk, t, [x, 0, 0])

        indsM = [i+1, j+1, k+1]
        qEndM = qEnd[:, :, indsM[0], indsM[1], indsM[2]]
        for d in range(ndim):
            indsL = indsM.copy()
            indsR = indsM.copy()
            indsL[d] -= 1
            indsR[d] += 1
            qEndL = qEnd[:, :, indsL[0], indsL[1], indsL[2]]
            qEndR = qEnd[:, :, indsR[0], indsR[1], indsR[2]]
            F[d,i,j,k] = interface(qEndL, qEndM, qEndR, d)

    for d in range(ndim):
        S -= F[d]/dxi[d]

    return dt * S
