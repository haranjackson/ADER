from numpy import array, zeros
from numpy.linalg import det

from models.nonlinear_elasticity.eos import total_energy
from models.nonlinear_elasticity.matrices import inv3


def Cvec(F, S, v, MP):
    """ Returns the vector of conserved variables, given the hyperelastic
        variables
    """
    Q = zeros(13)

    ρ = MP['ρ0'] / det(F)
    A = inv3(F)
    E = total_energy(A, S, v, MP)

    Q[:3] = ρ * v
    Q[3:12] = ρ * F.ravel()
    Q[12] = ρ * E

    return Q


MP = {
    'ρ0': 8.9,
    'α': 1,
    'β': 3,
    'γ': 2,
    'cv': 4e-4,
    'T0': 300,
    'b0': 2.1,
    'c0': 4.6
}

MP['K0'] = MP['c0']**2 - 4 / 3 * MP['b0']**2
MP['B0'] = MP['b0']**2


def five_wave_IC():

    tf = 0.06
    nx = 500

    dX = [1 / nx]

    vL = array([0, 0.5, 1])
    FL = array([[0.98, 0, 0], [0.02, 1, 0.1], [0, 0, 1]])
    SL = 0.001

    vR = array([0, 0, 0])
    FR = array([[1, 0, 0], [0, 1, 0.1], [0, 0, 1]])
    SR = 0

    QL = Cvec(FL, SL, vL, MP)
    QR = Cvec(FR, SR, vR, MP)

    u = zeros([nx, 13])

    for i in range(nx):
        if i / nx < 0.5:
            u[i] = QL
        else:
            u[i] = QR

    return u, MP, tf, dX


def seven_wave_IC():

    tf = 0.06
    nx = 500

    dX = [1 / nx]

    vL = array([2, 0, 0.1])
    FL = array([[1, 0, 0], [-0.01, 0.95, 0.02], [-0.015, 0, 0.9]])
    SL = 0

    vR = array([0, -0.03, -0.01])
    FR = array([[1, 0, 0], [0.015, 0.95, 0], [-0.01, 0, 0.9]])
    SR = 0

    QL = Cvec(FL, SL, vL, MP)
    QR = Cvec(FR, SR, vR, MP)

    u = zeros([nx, 13])

    for i in range(nx):
        if i / nx < 0.5:
            u[i] = QL
        else:
            u[i] = QR

    return u, MP, tf, dX
