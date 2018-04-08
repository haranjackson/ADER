from numpy import abs, amax, zeros
from numpy.linalg import eigvals
from tangent import autodiff


def make_M(F, B, NV):
    """ Returns a function that calculates the system Jacobian, given the flux
        function, and the non-conservative matrix (if necessary)
    """
    dFdQ = autodiff(F)

    def M(Q, d, pars=None):
        """ Returns the system Jacobian in direction d, given state Q
        """
        ret = zeros([NV, NV])
        for i in range(NV):
            x = zeros(NV)
            x[i] = 1
            ret[i] = dFdQ(Q, d, pars, x)
        if B is not None:
            ret += B(Q, d, pars)
        return ret

    return M


def make_max_eig(M):

    def max_eig(Q, d, pars=None):
        return amax(abs(eigvals(M(Q, d, pars))))

    return max_eig
