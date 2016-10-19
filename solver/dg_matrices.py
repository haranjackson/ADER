from itertools import product

from numpy import concatenate, diag, eye, ones, zeros
from scipy.sparse import csc_matrix

from options import ndim, N, NT, n
from auxiliary.basis import quad, basis_polys, derivative_values
from auxiliary.functions import kron_prod


def inner_products(nodes, weights, ψ, ψDer):
    """ Returns the elements of the matrices used in the Galerkin predictor
    """
    I11 = zeros([N+1, N+1])       # I11[a,b] = ψ_a(1) * ψ_b(1)
    I1 = zeros([N+1, N+1])        # I1[a,b] = <ψ_a, ψ_b>
    I2 = zeros([N+1, N+1])        # I2[a,b] = <ψ_a, ψ_b'>

    for a, b in product(range(N+1), range(N+1)):

        I11[a,b] = ψ[a](1) * ψ[b](1)

        if a==b:
            I1[a,b] = weights[a]
            I2[a,b] = (ψ[a](1)**2 - ψ[a](0)**2) / 2
        else:
            I2[a,b] = weights[a] * ψDer[1][b](nodes[a])

    return I11, I1, I2, eye(N+1)

def system_matrices():
    """ Returns the matrices used in the Galerkin predictor
    """
    nodes, _, weights = quad()
    ψ, ψDer, _        = basis_polys()
    derivs            = derivative_values()

    I11, I1, I2, I = inner_products(nodes, weights, ψ, ψDer)

    W = concatenate([ψ[a](0) * kron_prod([I1]*ndim) for a in range(N+1)])

    U = kron_prod([I11-I2.T] + [I1]*ndim)
    U = csc_matrix(U)

    V = zeros([ndim, NT, NT])
    for i in range(1, ndim+1):
        V[i-1] = kron_prod([I1]*i + [I2] + [I1]*(ndim-i))

    Z = kron_prod([I1]*(ndim+1))
    Z = (diag(Z) * ones([n, NT])).T

    T = zeros([ndim, NT, NT])
    for i in range(1, ndim+1):
        T[i-1] = kron_prod([I]*i + [derivs] + [I]*(ndim-i))

    return W, U, V, Z, T
