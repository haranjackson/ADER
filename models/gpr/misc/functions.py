from numpy import dot, eye, sqrt


def sigma_norm(σ):
    """ Returns the norm defined in Boscheri et al
    """
    tmp1 = (σ[0, 0] - σ[1, 1])**2 + \
           (σ[1, 1] - σ[2, 2])**2 + \
           (σ[2, 2] - σ[0, 0])**2

    tmp2 = σ[0, 1]**2 + σ[1, 2]**2 + σ[2, 0]**2

    return sqrt(0.5 * tmp1 + 3 * tmp2)


def dev(G):
    """ Returns the deviator of G
    """
    return G - G.trace() / 3 * eye(3)


def gram(A):
    """ Returns the Gram matrix for A
    """
    return dot(A.T, A)
