from numpy import dot, outer, sqrt, zeros
from numpy.linalg import eigvals

from example.gpr.variables.functions import dev, gram
from example.gpr.variables.eos import dTdρ, dTdp
from example.gpr.variables.state import pressure, temperature
from example.gpr.variables.wavespeeds import c_0, c_h
from example.parameters import cs, γ, α, cv


def max_abs_eigs(Q, d):
    """ Returns the maximum of the absolute values of the eigenvalues of the GPR system
    """
    O = zeros([4,4])

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].reshape([3,3])
    J = Q[14:17] / ρ

    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)

    vd = v[d]
    G = gram(A)
    Gd = G[d]

    O_ = dot(G, dev(G))
    O_[:, d] *= 2
    O_[d] *= 2
    O_[d, d] *= 3/4
    O_ += Gd[d] * G + 1/3 * outer(Gd, Gd)
    O_ *= cs**2
    O[:3, :3] = O_

    c0 = c_0(ρ, p)
    O[d, d] += c0**2

    Tρ = dTdρ(ρ, p)
    Tp = dTdp(ρ)
    ch = c_h(ρ, T)

    O[3, 0] = Tρ + Tp * c0**2
    O[0, 3] = ch**2 / Tp
    O[3, 3] = ch**2

    lam = sqrt(eigvals(O).max())

    if vd > 0:
        return vd + lam
    else:
        return lam - vd
