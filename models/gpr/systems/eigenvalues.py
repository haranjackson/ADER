from numpy import zeros

from models.gpr.variables.wavespeeds import c_0, c_h


def Xi1(P, d):

    ρ = P.ρ
    MP = P.MP
    ret = zeros([4, 5])

    ret[0, 1] = 1 / ρ

    dσdρ = P.dσdρ()
    dσdA = P.dσdA()
    ret[:3, 0] = -1 / ρ * dσdρ[d]
    ret[:3, 2:] = -1 / ρ * dσdA[d, :, :, d]

    if MP.THERMAL:
        dTdρ = P.dTdρ()
        dTdp = P.dTdp()
        ret[3, 0] = dTdρ / ρ
        ret[3, 1] = dTdp / ρ

    return ret


def Xi2(P, d):

    ρ = P.ρ
    p = P.p()
    A = P.A
    MP = P.MP
    c0 = c_0(ρ, p, A, MP)

    ret = zeros([5, 4])

    ret[0, 0] = ρ
    ret[1, d] = ρ * c0**2

    σ = P.σ()
    dσdρ = P.dσdρ()
    ret[1, :3] += σ[d] - ρ * dσdρ[d]
    ret[2:, :3] = A

    if MP.THERMAL:
        T = P.T()
        dTdp = P.dTdp()
        ch = c_h(ρ, T, MP)
        ret[1, 3] = ρ * ch**2 / dTdp

    return ret
