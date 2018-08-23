from numpy import ceil, floor, polyint, zeros


def weno_matrices(N, ψ):
    """ Returns a list of the matrices used in the WENO method
    """
    M = zeros([4, N, N])
    fN = int(floor((N - 1) / 2))
    cN = int(ceil((N - 1) / 2))

    for e in range(N):
        for p in range(N):
            P = polyint(ψ[p])
            M[0, e, p] = P(e - N + 2) - P(e - N + 1)
            M[1, e, p] = P(e + 1) - P(e)
            M[2, e, p] = P(e - cN + 1) - P(e - cN)
            M[3, e, p] = P(e - fN + 1) - P(e - fN)
    return M


def oscillation_indicator(N, dψ):

    Σ = zeros([N, N])
    for p in range(N):
        for m in range(N):
            for a in range(1, N):
                dψa = dψ[a]
                P = polyint(dψa[p] * dψa[m])
                Σ[p, m] += P(1) - P(0)
    return Σ
