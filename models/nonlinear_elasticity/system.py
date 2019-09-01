from numpy import sqrt, zeros
from numpy.linalg import det

from models.nonlinear_elasticity.state import entropy, sigma


def F_nonlinear_elasticity(Q, d, MP):

    ret = zeros(13)

    ρv = Q[:3]
    ρF = array([[Q[3], Q[4], Q[5]], [Q[6], Q[7], Q[8]], [Q[9], Q[10], Q[11]]])
    ρE = Q[12]

    ρ0 = MP['ρ0']
    ρ = sqrt(det(ρF) / ρ0)
    v = ρv / ρ

    F = array([[Q[3] / ρ, Q[4] / ρ, Q[5] / ρ], [Q[6] / ρ, Q[7] / ρ, Q[8] / ρ],
               [Q[9] / ρ, Q[10] / ρ, Q[11] / ρ]])

    E = ρE / ρ

    vd = ρv[d] / ρ
    S = entropy(E, F, v, MP)
    #σ = sigma(ρ, F, S, MP)

    #ret[:3] = vd * ρv - σ[d]
    #ret[3:6] = vd * ρF[0] - v[0] * ρF[d]
    #ret[6:9] = vd * ρF[1] - v[1] * ρF[d]
    #ret[9:12] = vd * ρF[2] - v[2] * ρF[d]
    #ret[12] = vd * ρE - dot(σ[d], v)

    return ret


def B_nonlinear_elasticity(Q, d, MP):

    ret = zeros([13, 13])

    ρv = Q[:3]
    ρF = Q[3:12].reshape([3, 3])

    ρ0 = MP['ρ0']
    ρ = sqrt(det(ρF) / ρ0)
    v = ρv / ρ

    for i in range(3):
        for j in range(3):
            ret[3 * (i + 1) + j, 3 * (d + 1) + j] = v[i]

    return ret
