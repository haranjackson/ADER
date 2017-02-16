from itertools import product

from numpy import absolute, array, dot, isnan, zeros
from scipy.sparse.linalg import spsolve
from scipy.optimize import newton_krylov

from options import stiff, superStiff, hidalgo, TOL, ndim, dxi, MAX_ITER, N, NT, n
from system import source, jacobian, flux, block
from dg_matrices import system_matrices
from auxiliary.basis import quad, derivative_values


W, U, V, Z, T = system_matrices()
_, gaps, _ = quad()
derivs = derivative_values()


def rhs(q, Ww, dt):
    """ Returns the right handside of the linear system governing the coefficients of qh
    """
    Tq = dot(T, q)
    Fq = zeros([ndim, NT, n])
    ret = zeros([NT, n])

    for b in range(NT):
        ret[b] = source(q[b])
        for d in range(ndim):
            ret[b] -= dot(block(q[b],d), Tq[d,b]) / dxi[d]
    ret *= Z

    for b in range(NT):
        for d in range(ndim):
            Fq[d,b] = flux(q[b], d)
    for d in range(ndim):
        ret -= dot(V[d], Fq[d]) / dxi[d]

    return dt * ret + Ww

def standard_initial_guess(w):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    q = array([w for i in range(N+1)])
    return q.reshape([NT, n])

def hidalgo_initial_guess(w, dtgaps):
    """ Returns the initial guess found in DOI: 10.1007/s10915-010-9426-6
        NB: Only implemented in 1D
    """
    dx = dxi[0]
    q = zeros([N+1]*(ndim+1) + [n])
    qj = w                          # Set values at start of time step to WENO reconstruction
    for j in range(N+1):            # Loop over the time levels in the DG predictor
        dt = dtgaps[j]
        dqdxj = dot(derivs, qj)

        for i in range(N+1):        # Loop over spatial nodes at time level j
            qij = qj[i]
            dqdxij = dqdxj[i]
            J = dot(jacobian(qij, 0), dqdxij)
            Sj = source(qij)

            if superStiff:          # If sources are very stiff, root must be found implicitly
                f = lambda X: X - qij + dt/dx * J - dt/2 * (Sj + source(X))
                q[j,i] = newton_krylov(f, qij, f_tol=TOL)
            else:
                q[j,i] = qij - dt/dx * J + dt * Sj

        qj = q[j]

    return q.reshape([NT, n])

def predictor(wh, dt):
    """ Returns the Galerkin predictor, given the WENO reconstruction at tn
    """
    nx, ny, nz, = wh.shape[:3]
    wh = wh.reshape([nx, ny, nz, (N+1)**ndim, n])
    qh = zeros([nx, ny, nz, NT, n])

    for i, j, k in product(range(nx), range(ny), range(nz)):

        w = wh[i, j, k]
        Ww = dot(W, w)

        if hidalgo:
            q = hidalgo_initial_guess(w, dt*gaps)
        else:
            q = standard_initial_guess(w)

        if stiff:
            func = lambda X: X - spsolve(U, rhs(X, Ww, dt))
            qh[i, j, k] = newton_krylov(func, q, f_tol=TOL, method='bicgstab')

        else:
            for count in range(MAX_ITER):
                qNew = spsolve(U, rhs(q, Ww, dt))

                if isnan(qNew).any():
                    print("DG root finding failed")
                    break
                elif (absolute(q-qNew) > TOL * (1 + absolute(q))).any():    # Check convergence
                    q = qNew
                    continue
                else:
                    qh[i, j, k] = qNew
                    break
            else:
                print("Maximum iterations reached")
    return qh
