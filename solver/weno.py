""" Implements the WENO method used in Dumbser et al (DOI 10.1016/j.cma.2013.09.022)
"""
from itertools import product

from numpy import ceil, dot, einsum, floor, zeros, multiply as mult
from scipy.linalg import solve

from options import rc, λc, λs, ε, ndim, N, n
from .weno_matrices import coefficient_matrices, oscillation_indicator
from auxiliary.functions import extend


ML,MR,MCL,MCR = coefficient_matrices()
Σ = oscillation_indicator()
fhN = int(floor(N/2))
chN = int(ceil(N/2))
fhN2 = fhN+N+1
chN2 = chN+N+1


if N%2:
    nStencils = 4
else:
    nStencils = 3


def weights(w, λ):
    Σw  = dot(Σ, w)
    σ  = einsum('ki,ki->i', w, Σw)
    return λ / (abs(σ) + ε)**rc

def coeffs(u1):
    """ Calculate coefficients of basis polynomials and weights
    """
    wL  = solve(ML, u1[:N+1])
    wR  = solve(MR, u1[N:])
    oL  = weights(wL, λs)
    oR  = weights(wR, λs)
    if N==1:
        return (mult(wL,oL) + mult(wR,oR)) / (oL + oR)

    wCL = solve(MCL, u1[fhN:fhN2])
    oCL = weights(wCL, λc)
    if nStencils==3:
        return (mult(wL,oL) + mult(wCL,oCL) + mult(wR,oR)) / (oL + oCL + oR)

    oCR = weights(wCR, λc)
    wCR = solve(MCR, u1[chN:chN2])
    return (mult(wL,oL) + mult(wCL,oCL) + mult(wCR,oCR) + mult(wR,oR)) / (oL + oCL + oCR + oR)

def reconstruct(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]
    c = 2*N+1

    Wx = zeros([nx, ny, nz, N+1, n])
    tempW = extend(u, N, 0)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        Wx[i, j, k] = coeffs(tempW[i:i+c, j, k])
    if ndim==1:
        return Wx

    Wxy = zeros([nx, ny, nz, N+1, N+1, n])
    tempWx = extend(Wx, N, 1)
    for i, j, k, a in product(range(nx), range(ny), range(nz), range(N+1)):
        Wxy[i, j, k, a] = coeffs(tempWx[i, j:j+c, k, a])
    if ndim==2:
        return Wxy

    Wxyz = zeros([nx, ny, nz, N+1, N+1, N+1, n])
    tempWxy = extend(Wxy, N, 2)
    for i, j, k, a, b in product(range(nx), range(ny), range(nz), range(N+1), range(N+1)):
        Wxyz[i, j, k, a, b] = coeffs(tempWxy[i, j, k:k+c, a, b])
    return Wxyz
