from itertools import product

from numpy import array, dot, zeros
from scipy.optimize import newton_krylov

from ader.etc.basis import derivative


def standard_initial_guess(obj, w, *args):
    """ Returns a Galerkin intial guess consisting of the value of q at t=0
    """
    ret = array([w for i in range(obj.N)])
    return ret.reshape(obj.NT, obj.NV)


def stiff_initial_guess(obj, w, dt, dX):
    """ Returns an initial guess based on the underlying equations
    """
    q = zeros([obj.N] * (obj.NDIM + 1) + [obj.NV])
    qt = w.reshape([obj.N] * obj.NDIM + [obj.NV])
    indList = [range(obj.N)] * obj.NDIM

    for t in range(obj.N):

        dt_ = dt * obj.basis.GAPS[t]

        # loop over the indices of each spatial node
        for inds in product(*indList):

            q_ = qt[inds]     # the value of q at the current spatial node

            Mdqdx = zeros(obj.NV)
            for d in range(obj.NDIM):
                dqdx = derivative(obj.N, obj.NV, obj.NDIM, qt, inds, d,
                                  obj.basis.DERVALS)
                Mdqdx += dot(obj.system_matrix(q_, d, obj.model_params), dqdx) / dX[d]

            S0 = obj.source(q_, obj.model_params)

            if obj.nk_ig:

                def f(X):

                    S = (S0 + obj.source(X, obj.model_params)) / 2
                    return X - q_ + dt_ * (Mdqdx - S)

                q[(t,) + inds] = newton_krylov(f, q_, f_tol=obj.tol)

            else:
                q[(t,) + inds] = q_ - dt_ * (Mdqdx - S0)

        qt = q[t]

    return q.reshape(obj.NT, obj.NV)
