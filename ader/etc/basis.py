from types import SimpleNamespace

from numpy import concatenate, dot, eye, polyder, zeros
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import lagrange


def gauss_quad(N):
    """ NODES, WGHTS are the Legendre-Gauss nodes and weights, scaled to [0,1]
        GAPS contains the gaps between successive nodes
    """
    NODES, WGHTS = leggauss(N)
    NODES += 1
    NODES /= 2
    WGHTS /= 2
    GAPS = NODES - concatenate(([0], NODES[:-1]))
    return NODES, WGHTS, GAPS


def basis_polys(N, NODES):
    """ ψ contains the basis polynomials
        dψ[i,j] is the ith derivative of the jth basis polynomial
    """
    ψ = [lagrange(NODES, eye(N)[i]) for i in range(N)]
    dψ = [[polyder(p, m=i) for p in ψ] for i in range(N + 1)]
    return ψ, dψ


def basis_vals(N, NODES, ψ, dψ):
    """ ENDVALS is the value of the ith basis function at j=0 and j=1
        DERVALS is the value of the derivative of the jth basis function at the
        ith node
    """
    ENDVALS = zeros([N, 2])
    for i in range(N):
        ENDVALS[i, 0] = ψ[i](0)
        ENDVALS[i, 1] = ψ[i](1)

    DERVALS = zeros([N, N])
    for i in range(N):
        for j in range(N):
            DERVALS[i, j] = dψ[1][j](NODES[i])

    return ENDVALS, DERVALS


def Basis(N):
    basis = SimpleNamespace()
    basis.NODES, basis.WGHTS, basis.GAPS = gauss_quad(N)
    basis.ψ, basis.dψ = basis_polys(N, basis.NODES)
    basis.ENDVALS, basis.DERVALS = basis_vals(N, basis.NODES, basis.ψ, basis.dψ)
    return basis


def flat_index(inds, N):
    """ If inds = (i1,...,in) is the index of an element in a hypercube of N^n
        points, then this function returns the corresponding index when the
        hypercube is represented by a 1d array
    """
    if len(inds) == 0:
        return 0
    elif len(inds) == 1:
        return inds[0]
    else:
        return N * flat_index(inds[:-1], N) + inds[-1]


def derivative(N, NV, NDIM, q, inds, d, DERVALS):
    """ Returns derivative of q at the point given by inds in direction d
    """
    i = flat_index(inds[:d], N)
    j = flat_index(inds[d + 1:], N)
    qx = q.reshape(N**d, N, N**(NDIM - d - 1), NV)[i, :, j]
    return dot(DERVALS[inds[d]], qx)
