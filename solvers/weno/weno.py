""" Implements WENO method used in Dumbser et al, DOI 10.1016/j.cma.2013.09.022
"""
from itertools import product

from numpy import ceil, concatenate, floor, int64, multiply, ones, zeros
from scipy.linalg import solve

from solvers.misc import extend
from solvers.weno.matrices import fHalfN, cHalfN, WN_M, WN_Σ
from options import rc, λc, λs, ε, ndim, N, nV


LAMS = [λs, λs, λc, λc]


def calculate_coeffs(uList):
    """ Calculate coefficients of basis polynomials and weights
    """
    n = len(uList)
    wList = [solve(WN_M[i], uList[i], overwrite_b=1, check_finite=0)
             for i in range(n)]

    σList = [((w.T).dot(WN_Σ).dot(w)).diagonal() for w in wList]
    oList = [LAMS[i] / (abs(σList[i]) + ε)**rc for i in range(n)]
    oSum = zeros(nV)
    numerator = zeros([N, nV])
    for i in range(n):
        oSum += oList[i]
        numerator += multiply(wList[i], oList[i])
    return numerator / oSum


def coeffs(ret, uL, uR, uCL, uCR):

    if N == 2:
        ret[:] = calculate_coeffs([uL, uR])
    elif N % 2:
        ret[:] = calculate_coeffs([uL, uR, uCL])
    else:
        ret[:] = calculate_coeffs([uL, uR, uCL, uCR])


def extract_stencils(arrayRow, ind):
    uL = arrayRow[ind - N + 1: ind + 1]
    uR = arrayRow[ind: ind + N]
    uCL = arrayRow[ind - cHalfN: ind + fHalfN + 1]
    uCR = arrayRow[ind - fHalfN: ind + cHalfN + 1]
    return uL, uR, uCL, uCR


def weno_launcher(u):
    """ Find reconstruction coefficients of u to order N+1
    """
    nx, ny, nz = u.shape[:3]

    Wx = zeros([nx, ny, nz, N, nV])
    tempu = extend(u, N - 1, 0)
    for i, j, k in product(range(nx), range(ny), range(nz)):
        uL, uR, uCL, uCR = extract_stencils(tempu[:, j, k], i + N - 1)
        coeffs(Wx[i, j, k], uL, uR, uCL, uCR)

    if ny == 1:
        return Wx

    Wxy = zeros([nx, ny, nz, N, N, nV])
    tempWx = extend(Wx, N - 1, 1)
    for i, j, k, a in product(range(nx), range(ny), range(nz), range(N)):
        uL, uR, uCL, uCR = extract_stencils(tempWx[i, :, k, a], j + N - 1)
        coeffs(Wxy[i, j, k, a], uL, uR, uCL, uCR)

    if nz == 1:
        return Wxy

    Wxyz = zeros([nx, ny, nz, N, N, N, nV])
    tempWxy = extend(Wxy, N - 1, 2)
    for i, j, k, a, b in product(range(nx), range(ny), range(nz), range(N),
                                 range(N)):
        uL, uR, uCL, uCR = extract_stencils(tempWxy[i, j, :, a, b], k + N - 1)
        coeffs(Wxyz[i, j, k, a, b], uL, uR, uCL, uCR)

    return Wxyz
