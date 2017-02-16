from numpy import concatenate, eye, polyder, polyint, zeros
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange

from options import N


def quad():
    """ Returns the Legendre-Gauss nodes and weights, scaled to [0,1]
    """
    nodes, weights = leggauss(N+1)
    nodes += 1
    nodes /= 2
    weights /= 2
    gaps = nodes - concatenate(([0], nodes[:-1]))
    return nodes, gaps, weights

def basis_polys():
    """ Returns the basis polynomials and their derivatives and antiderivatives
    """
    nodes, _, _ = quad()
    ψ = [lagrange(nodes,eye(N+1)[i]) for i in range(N+1)]
    ψDer = [[polyder(ψ_p, m=a) for ψ_p in ψ] for a in range(N+1)]
    ψInt = [polyint(ψ_p) for ψ_p in ψ]
    return ψ, ψDer, ψInt

def end_values():
    """ Returns the values of the basis functions at 0 and 1
    """
    ψ, _, _ = basis_polys()
    ret = zeros([N+1, 2])
    for i in range(N+1):
        ret[i,0] = ψ[i](0)
        ret[i,1] = ψ[i](1)
    return ret

def derivative_values():
    """ Returns the value of the derivative of the jth basis function at the ith node
    """
    nodes, _, _ = quad()
    _, psiDer, _ = basis_polys()
    ret = zeros([N+1, N+1])
    for i in range(N+1):
        for j in range(N+1):
            ret[i,j] = psiDer[1][j](nodes[i])
    return ret
