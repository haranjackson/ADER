from itertools import product

from joblib import delayed
from numpy import absolute, array, concatenate, dot, zeros
from scipy.linalg import solve
from scipy.optimize import newton_krylov

from solvers.dg.matrices import DG_W, DG_U, DG_V, DG_Z, DG_T
from system import flux, block, source
from options import ndim, dx, N, NT, nV, STIFF, DG_TOL, DG_IT, PARA_DG, NCORE


def rhs(q, Ww, dt):
    """ Returns the right-handside of the system governing coefficients of qh
    """
    ret = zeros([NT, nV])

    Tq = dot(DG_T, q)
    Fq = zeros([ndim, NT, nV])
    Bq = zeros([ndim, NT, nV])
    B = zeros([nV, nV])
    for b in range(NT):
        qb = q[b]
        source(ret[b], qb)
        for d in range(ndim):
            flux(Fq[d, b], qb, d)
            block(B, qb, d)
            Bq[d, b] = dot(B, Tq[d, b])

    ret *= dx

    for d in range(ndim):
        ret -= Bq[d]

    ret *= DG_Z
    for d in range(ndim):
        ret -= dot(DG_V[d], Fq[d])

    return (dt / dx) * ret + Ww


def initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(N)])
    return ret.reshape([NT, nV])


def unconverged(q, qNew):
    """ Mixed convergence condition
    """
    return (absolute(q - qNew) > DG_TOL * (1 + absolute(q))).any()


def predictor(wh, dt):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    nx, ny, nz, = wh.shape[:3]
    wh = wh.reshape([nx, ny, nz, N**ndim, nV])
    qh = zeros([nx, ny, nz, NT, nV])

    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        Ww = dot(DG_W, w)

        def obj(X): return dot(DG_U, X) - rhs(X, Ww, dt)

        q = initial_guess(w)

        if STIFF:
            qh[i, j, k] = newton_krylov(obj, q, f_tol=DG_TOL, method='bicgstab')

        else:

            for count in range(DG_IT):

                qNew = solve(DG_U, rhs(q, Ww, dt),
                             check_finite=False)

                if unconverged(q, qNew):
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break

            # revert to stiff solver
            else:
                q = initial_guess(w)
                qh[i, j, k] = newton_krylov(
                    obj, q, f_tol=DG_TOL, method='bicgstab')

    return qh


def dg_launcher(pool, wh, dt):
    """ Controls the parallel computation of the Galerkin predictor
    """
    if PARA_DG:
        nx = wh.shape[0]
        step = int(nx / NCORE)
        chunk = array([i * step for i in range(NCORE)] + [nx + 1])
        n = len(chunk) - 1
        qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i + 1]], dt)
                      for i in range(n))
        return concatenate(qhList)
    else:
        return predictor(wh, dt)
