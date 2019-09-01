from numpy import array, eye, linspace, sqrt, zeros
from scipy.special import erf

from models.gpr.misc.objects import material_parameters
from models.gpr.misc.structures import Cvec


def riemann_IC(final_time, n, dX, ρL, pL, vL, ρR, pR, vR, model_params):
    """ Constructs the riemann problem corresponding to the parameters given
    """
    AL = ρL**(1 / 3) * eye(3)
    JL = zeros(3)
    AR = ρR**(1 / 3) * eye(3)
    JR = zeros(3)

    QL = Cvec(ρL, pL, vL, AL, JL, model_params)
    QR = Cvec(ρR, pR, vR, AR, JR, model_params)

    u = zeros([n, 17])

    for i in range(n):

        if i * dX[0] < 0.5:
            u[i] = QL
        else:
            u[i] = QR

    return u, model_params, final_time, dX


def first_stokes_problem_IC(μ=1e-2, n=100, v0=0.1, final_time=1):

    Lx = 1

    γ = 1.4

    ρL = 1
    pL = 1 / γ
    vL = array([0, -v0, 0])

    ρR = 1
    pR = 1 / γ
    vR = array([0, v0, 0])

    model_params = material_parameters(EOS='sg',
                                       ρ0=1,
                                       cv=1,
                                       p0=1 / γ,
                                       γ=γ,
                                       cs=1,
                                       cα=1e-16,
                                       μ=μ,
                                       Pr=0.75)
    dX = [Lx / n]

    return riemann_IC(final_time, n, dX, ρL, pL, vL, ρR, pR, vR, model_params)


def first_stokes_problem_exact(μ=1e-2, n=100, v0=0.1, t=1):
    """ Returns the exact solution of the y-velocity in the x-axis for Stokes'
        First Problem
    """
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))
