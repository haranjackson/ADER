from itertools import product

from joblib import delayed
from numpy import array, concatenate, dot, einsum, tensordot, zeros

from solvers.fv.fluxes import B_INT, D_RUS, D_ROE, D_OSH
from solvers.basis import WGHTS, ENDVALS, DERVALS
from system import block, source, flux
from options import ndim, nV, dx, N, FLUX, PARA_FV, NCORE


if FLUX == 0:
    D_FUN = D_RUS
elif FLUX == 1:
    D_FUN = D_ROE
elif FLUX == 2:
    D_FUN = D_OSH


WGHT_LIST = [WGHTS] * (ndim + 1) + [array([1])] * (3 - ndim)

WGHT_PROD = einsum('t,x,y,z', WGHT_LIST[0],
                   WGHT_LIST[1], WGHT_LIST[2], WGHT_LIST[3])
WGHT_PROD_END = einsum('t,x,y', WGHT_LIST[1], WGHT_LIST[2], WGHT_LIST[3])

IDX = [N] * (ndim + 1) + [1] * (3 - ndim)


def endpoints(qh):
    """ qEnd[d,e,i,j,k,:,:,:,:,:] is the set of polynomial coefficients in cell
        i,j,k at end e (0=left, 1=right) in the dth direction (0=xaxis, 1=yaxis,
        2=zaxis)
    """
    nx, ny, nz = qh.shape[:3]
    qh0 = qh.reshape([nx, ny, nz] + IDX + [nV])
    qEnd = zeros([ndim, 2, nx, ny, nz] + IDX[1:] + [nV])

    for d in range(ndim):
        qEnd[d] = tensordot(ENDVALS, qh0, (0, 4 + d))

    return qEnd


def interfaces(qEnd):
    """ Returns the interface terms in the Finite Volume update
    """
    nx, ny, nz = qEnd.shape[2:5]
    fEnd = zeros([ndim, nx - 1, ny - 1, nz - 1, nV])
    BEnd = zeros([ndim, nx - 1, ny - 1, nz - 1, nV])

    for d in range(ndim):
        for i, j, k in product(range(nx - 1), range(ny - 1), range(nz - 1)):

            qL = qEnd[d, 1, i, j, k]
            if d == 0:
                qR = qEnd[d, 0, i + 1, j, k]
            elif d == 1:
                qR = qEnd[d, 0, i, j + 1, k]
            else:
                qR = qEnd[d, 0, i, j, k + 1]

            fEndTemp = zeros(nV)
            BEndTemp = zeros(nV)
            for t, x1, x2 in product(range(IDX[1]), range(IDX[2]),
                                     range(IDX[3])):

                qL_ = qL[t, x1, x2]
                qR_ = qR[t, x1, x2]

                ftemp = zeros(nV)
                flux(ftemp, qL_, d)
                flux(ftemp, qR_, d)
                ftemp -= D_FUN(qL_, qR_, d)
                fEndTemp += WGHT_PROD_END[t, x1, x2] * ftemp
                BEndTemp += WGHT_PROD_END[t, x1, x2] * B_INT(qL_, qR_, d)

            fEnd[d, i, j, k] = fEndTemp
            BEnd[d, i, j, k] = BEndTemp

    ret = zeros([nx - 2, ny - 2, nz - 2, nV])
    ret -= fEnd[0, :-1, 1:, 1:]
    ret += fEnd[0, 1:,  1:, 1:]
    ret += BEnd[0, :-1, 1:, 1:]
    ret += BEnd[0, 1:,  1:, 1:]
    if ndim > 1:
        ret -= fEnd[1, 1:, :-1, 1:]
        ret += fEnd[1, 1:,  1:, 1:]
        ret += BEnd[1, 1:, :-1, 1:]
        ret += BEnd[1, 1:,  1:, 1:]
    if ndim > 2:
        ret -= fEnd[2, 1:, 1:, :-1]
        ret += fEnd[2, 1:, 1:,  1:]
        ret += BEnd[2, 1:, 1:, :-1]
        ret += BEnd[2, 1:, 1:,  1:]
    return ret


def center(qhi, t, inds):
    """ Returns the space-time averaged source term and non-conservative term
        in cell given by inds
    """
    q = qhi[t, inds[0], inds[1], inds[2]]
    qx = qhi[t, :, inds[1], inds[2]]
    qy = qhi[t, inds[0], :, inds[2]]
    qz = qhi[t, inds[0], inds[1], :]
    qi = [qx, qy, qz]

    ret = zeros(nV)

    source(ret, q)
    ret *= dx

    B = zeros([nV, nV])
    for d in range(ndim):
        dxdxi = dot(DERVALS[inds[d]], qi[d])
        block(B, q, d)
        ret -= dot(B, dxdxi)

    return ret


def fv_terms(qh, dt):
    """ Returns the space-time averaged interface terms, jump terms,
        source terms, and non-conservative terms
    """
    nx, ny, nz = qh.shape[:3]
    if nz == 1:
        qh0 = qh.repeat([3], axis=2)
    if ny == 1:
        qh0 = qh0.repeat([3], axis=1)
    nx, ny, nz = array(qh0.shape[:3]) - 2
    qEnd = endpoints(qh0)
    qh0 = qh0.reshape([nx + 2, ny + 2, nz + 2] + IDX + [nV])

    s = zeros([nx, ny, nz, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):

        qhi = qh0[i + 1, j + 1, k + 1]

        for t, x, y, z in product(range(IDX[0]), range(IDX[1]), range(IDX[2]),
                                  range(IDX[3])):

            s[i, j, k] += WGHT_PROD[t, x, y, z] * center(qhi, t, [x, y, z])

    s -= 0.5 * interfaces(qEnd)

    return dt / dx * s


def fv_launcher(pool, qh, dt):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    if PARA_FV:
        nx = qh.shape[0]
        step = int(nx / NCORE)
        chunk = array([i * step for i in range(NCORE)] + [nx + 1])
        chunk[0] += 1
        chunk[-1] -= 1
        n = len(chunk) - 1
        qhList = pool(delayed(fv_terms)(qh[chunk[i] - 1:chunk[i + 1] + 1], dt)
                      for i in range(n))
        return concatenate(qhList)
    else:
        return fv_terms(qh, dt)
