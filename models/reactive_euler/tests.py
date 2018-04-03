from numpy import array, concatenate, zeros

from models.reactive_euler.system import energy


def shock_induced_detonation_IC():

    MP = {'γ': 1.4,
          'cv': 2.5,
          'Qc': 1,
          'Ti': 0.25,
          'K0': 250}

    nx = 400

    ρL = 1.4
    pL = 1
    vL = array([0, 0, 0])
    λL = 0
    EL = energy(ρL, pL, vL, λL, MP['γ'], MP['Qc'])

    ρR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    λR = 1
    ER = energy(ρR, pR, vR, λR, MP['γ'], MP['Qc'])

    QL = ρL * concatenate([array([1, EL]), vL, array([λL])])
    QR = ρR * concatenate([array([1, ER]), vR, array([λR])])

    u = zeros([nx, 6])
    for i in range(nx):
        if i / nx < 0.25:
            u[i] = QL
        else:
            u[i] = QR

    tf = 0.5
    dX = [1 / nx]

    return u, MP, tf, dX
