from numpy import zeros
from scipy.linalg import eigvals

from options import n, ndim
from auxiliary.functions import extend


def flux(Q, d):
    """ Returns the flux vector in the dth direction
    """
    return zeros(n)

def block(Q, d):
    """ Returns the nonconvervative matrix in the dth direction
    """
    return zeros([n, n])

def source(Q):
    """ Returns the source vector
    """
    return zeros(n)

def jacobian(Q, d):
    """ Returns the Jacobian in the dth direction
    """
    return zeros([n, n])

def max_abs_eigs(q, d):
    """ Returns the largest of the absolute values of the eigenvalues of system matrix at q in
        direction d
    """
    J = jacobian(q, d)
    return max(abs(eigvals(J)))

def boundary_condition(u):
    """ Returns a copy of u with transmissive boundary conditions applied
    """
    ret = u.copy()
    for d in range(ndim):
        ret = extend(ret, 1, d)
    return ret
