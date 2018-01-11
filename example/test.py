from numpy import array, eye, zeros, arange, exp
from scipy.optimize import brentq

from example.gpr.variables.eos import total_energy
from example.gpr.variables.wavespeeds import c_0
from example.parameters import tf, Ms, γ, ρ0, p0, μ
from options import nx, nV, Lx


def Cvec(ρ, p, v, A, J):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(nV)

    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()
    Q[14:17] = ρ * J

    return Q

def viscous_shock_exact(x, μ):
    """ Returns the density, pressure, and velocity of the viscous shock
        (Mach number Ms) at x
    """
    l = 0.3

    if x > l:
        x=l
    elif x < -l:
        x=-l

    c0 = c_0(ρ0, p0)
    a = 2 / (Ms**2 * (γ+1)) + (γ-1)/(γ+1)
    Re = ρ0 * c0 * Ms / μ
    c1 = ((1-a)/2)**(1-a)
    c2 = 3/4 * Re * (Ms**2-1) / (γ*Ms**2)

    f = lambda z: (1-z)/(z-a)**a - c1 * exp(c2*-x)

    vbar = brentq(f, a+1e-16, 1)
    p = p0 / vbar * (1 + (γ-1)/2 * Ms**2 * (1-vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0  - v    # Shock travelling into fluid at rest

    return ρ, p, v

def viscous_shock_IC():
    """ Test taken from DOI:10.1016/j.jcp.2016.02.015
    """
    x = arange(-Lx/2, Lx/2, 1/nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i], μ)

    tmp = zeros([nx, 1, 1, nV])
    for i in range(nx):
        A = (ρ[i])**(1/3) * eye(3)
        J = zeros(3)
        tmp[i,0,0] = Cvec(ρ[i], p[i], array([v[i], 0, 0]), A, J)

    u = zeros([nx, 1, 1, nV])
    for i in range(nx):
        ind = int(i + 0.3*nx)
        if ind >= nx:
            ind = nx - 1
        u[i] = tmp[ind]

    return u, tf
