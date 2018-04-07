from numpy import zeros


def sqnorm(x):
    """ Returns the squared norm of a 3-vector
        NOTE: numpy.linalg.norm and numpy.dot are not currently supported by
        the tangent library, so this function is used instead
    """
    return x[0]**2 + x[1]**2 + x[2]**2


def pressure(ρ, E, v, λ, γ, Qc):
    e = E - sqnorm(v) / 2 - Qc * (λ - 1)
    return (γ - 1) * ρ * e


def temperature(ρ, E, v, λ, Qc, cv):
    e = E - sqnorm(v) / 2 - Qc * (λ - 1)
    return e / cv


def energy(ρ, p, v, λ, γ, Qc):
    return p / ((γ - 1) * ρ) + sqnorm(v) / 2 + Qc * (λ - 1)


def F_reactive_euler(Q, d, model_params):

    γ = model_params['γ']
    Qc = model_params['Qc']

    ρ = Q[0]
    E = Q[1] / ρ
    v = [Q[2] / ρ, Q[3] / ρ, Q[4] / ρ]
    λ = Q[5] / ρ

    p = pressure(ρ, E, v, λ, γ, Qc)

    ret = v[d] * Q
    ret[1] += p * v[d]
    ret[2 + d] += p

    return ret


def reaction_rate(ρ, E, v, λ, Qc, cv, Ti, K0):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    T = temperature(ρ, E, v, λ, Qc, cv)
    return K0 if T > Ti else 0


def S_reactive_euler(Q, model_params):

    ret = zeros(6)

    Qc = model_params['Qc']
    cv = model_params['cv']
    Ti = model_params['Ti']
    K0 = model_params['K0']

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    λ = Q[5] / ρ

    ret[5] = -ρ * λ * reaction_rate(ρ, E, v, λ, Qc, cv, Ti, K0)

    return ret
