from numpy import zeros

from models.gpr.misc.functions import gram
from models.gpr.variables.derivatives import dEdρ, dEdp, dEdA, dEdJ, dTdρ, dTdp
from models.gpr.variables.eos import total_energy
from models.gpr.variables.sources import theta1inv, theta2inv
from models.gpr.variables.state import heat_flux, pressure, temperature
from models.gpr.variables.state import sigma, dsigmadρ, dsigmadA


def calculate_pressure(state):
    if hasattr(state, 'p_'):
        return state.p_
    else:
        if state.MP.THERMAL:
            state.p_ = pressure(state.ρ, state.E, state.v, state.A, state.J,
                                state.MP)
        else:
            state.p_ = pressure(state.ρ, state.E, state.v, state.A, zeros(3),
                                state.MP)
        return state.p_


class State():
    """ Returns the primitive varialbes, given a vector of conserved variables
    """

    def __init__(self, Q, MP):

        self.ρ = Q[0]
        self.E = Q[1] / self.ρ
        self.v = Q[2:5] / self.ρ
        self.A = Q[5:14].reshape([3, 3])

        if MP.THERMAL:
            self.J = Q[14:17] / self.ρ

        self.MP = MP

    def G(self):
        return gram(self.A)

    def p(self):
        return calculate_pressure(self)

    def T(self):
        if hasattr(self, 'T_'):
            return self.T_
        else:
            self.T_ = temperature(self.ρ, self.p(), self.MP)
            return self.T_

    def σ(self):
        return sigma(self.ρ, self.A, self.MP)

    def dσdρ(self):
        return dsigmadρ(self.ρ, self.A, self.MP)

    def dσdA(self):
        return dsigmadA(self.ρ, self.A, self.MP)

    def q(self):
        return heat_flux(self.T(), self.J, self.MP)

    def dEdρ(self):
        return dEdρ(self.ρ, self.p(), self.A, self.MP)

    def dEdp(self):
        return dEdp(self.ρ, self.MP)

    def ψ(self):
        return dEdA(self.ρ, self.A, self.MP)

    def H(self):
        return dEdJ(self.J, self.MP)

    def dTdρ(self):
        return dTdρ(self.ρ, self.p(), self.MP)

    def dTdp(self):
        return dTdp(self.ρ, self.MP)

    def θ1_1(self):
        return theta1inv(self.ρ, self.A, self.MP)

    def θ2_1(self):
        return theta2inv(self.ρ, self.T(), self.MP)


def Cvec(ρ, p, v, A, J, MP):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(17)
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J, MP)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()
    if MP.THERMAL:
        Q[14:17] = ρ * J

    return Q
