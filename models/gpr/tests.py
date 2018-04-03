from numpy import array, eye, arange, exp, linspace, sqrt, zeros
from scipy.optimize import brentq
from scipy.special import erf

from ader.etc.boundaries import standard_BC
from models.gpr.misc.objects import material_parameters
from models.gpr.misc.structures import Cvec
from models.gpr.variables.wavespeeds import c_0


def cell_sizes(Lx, nx, Ly=1, ny=1, Lz=1, nz=1):
    return array([Lx / nx, Ly / ny, Lz / nz])


def riemann_IC(tf, nx, dX, QL, QR, MP):

    u = zeros([nx, 17])

    for i in range(nx):

        if i * dX[0] < 0.5:
            u[i] = QL
        else:
            u[i] = QR

    return u, MP, tf, dX


def fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP):
    """ constructs the riemann problem corresponding to the parameters given
    """
    AL = ρL**(1 / 3) * eye(3)
    JL = zeros(3)
    AR = ρR**(1 / 3) * eye(3)
    JR = zeros(3)

    QL = Cvec(ρL, pL, vL, AL, JL, MP)
    QR = Cvec(ρR, pR, vR, AR, JR, MP)

    return riemann_IC(tf, nx, dX, QL, QR, MP)


def heat_conduction_IC():

    tf = 1
    nx = 100
    Lx = 1

    ρL = 2
    pL = 1
    vL = zeros(3)

    ρR = 0.5
    pR = 1
    vR = zeros(3)

    MP = material_parameters(EOS='sg', ρ0=1, cv=2.5, p0=1, γ=1.4, pINF=0,
                             cs=1, cα=2, μ=1e-2, κ=1e-2)
    dX = cell_sizes(Lx, nx)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP)


def first_stokes_problem_exact(μ, n=100, v0=0.1, t=1):
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))


def first_stokes_problem_IC():

    tf = 1
    nx = 200
    Lx = 1

    γ = 1.4
    μ = 1e-2  # 1e-3 # 1e-4

    ρL = 1
    pL = 1 / γ
    vL = array([0, -0.1, 0])

    ρR = 1
    pR = 1 / γ
    vR = array([0, 0.1, 0])

    MP = material_parameters(EOS='sg', ρ0=1, cv=1, p0=1 / γ, γ=γ, pINF=0,
                             cs=1, cα=1e-16, μ=μ, Pr=0.75)
    dX = cell_sizes(Lx, nx)

    return fluids_IC(tf, nx, dX, ρL, pL, vL, ρR, pR, vR, MP)


def viscous_shock_exact(x, Ms, MP, μ, center=0):
    """ Returns the density, pressure, and velocity of the viscous shock
        (Mach number Ms) at x
    """
    x -= center
    ρ0 = MP.ρ0
    p0 = MP.p0
    γ = MP.γ

    if Ms == 2:
        l = 0.3
    elif Ms == 3:
        l = 0.13

    if x > l:
        x = l
    elif x < -l:
        x = -l

    c0 = c_0(ρ0, p0, eye(3), MP)
    a = 2 / (Ms**2 * (γ + 1)) + (γ - 1) / (γ + 1)
    Re = ρ0 * c0 * Ms / μ
    c1 = ((1 - a) / 2)**(1 - a)
    c2 = 3 / 4 * Re * (Ms**2 - 1) / (γ * Ms**2)

    def f(z): return (1 - z) / (z - a)**a - c1 * exp(c2 * -x)

    vbar = brentq(f, a + 1e-16, 1)
    p = p0 / vbar * (1 + (γ - 1) / 2 * Ms**2 * (1 - vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0 - v    # Shock travelling into fluid at rest

    return ρ, p, v


def viscous_shock_IC(center=0):

    tf = 0.2
    nx = 100
    Lx = 1

    Ms = 2
    γ = 1.4
    ρ0 = 1
    p0 = 1 / γ
    μ = 2e-2

    MP = material_parameters(EOS='sg', ρ0=ρ0, cv=2.5, p0=p0, γ=γ, pINF=0,
                             cs=5, cα=5, μ=2e-2, Pr=0.75)
    dX = cell_sizes(Lx, nx)

    x = arange(-Lx / 2, Lx / 2, 1 / nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i], Ms, MP, μ, center=center)

    v -= v[0]           # Velocity in shock 0

    u = zeros([nx, 17])
    for i in range(nx):
        A = (ρ[i])**(1 / 3) * eye(3)
        J = zeros(3)
        u[i] = Cvec(ρ[i], p[i], array([v[i], 0, 0]), A, J, MP)

    return u, MP, tf, dX


def piston_IC():

    tf = 1.5
    nx = 300
    Lx = 1.5

    dX = cell_sizes(Lx, nx)

    MP = material_parameters(EOS='smg', ρ0=8.93, cv=1, p0=0,
                             c0=0.394, Γ0=2, s=1.48, e0=0,
                             cs=0.219, σY=9e-4, τ1=0.1, n=10, PLASTIC=True)
    ρ = MP.ρ0
    p = MP.p0
    v = zeros(3)
    A = eye(3)
    J = zeros(3)
    Q = Cvec(ρ, p, v, A, J, MP)

    u = zeros([nx, 17])

    for i in range(nx):
        u[i] = Q

    return u, MP, tf, dX


def piston_BC(u, N):
    ret = standard_BC(u)
    ret[:N, 2:5] = ret[N, 0] * array([0.002, 0, 0])
    return ret
