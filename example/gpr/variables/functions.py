from numpy import dot, eye, kron, trace


def gram(A):
    """ Returns the Gram matrix for A
    """
    return dot(A.T, A)


def kron_prod(matList):
    """ Returns the kronecker product of the matrices in matList
    """
    ret = matList[0]
    for i in range(1, len(matList)):
        ret = kron(ret, matList[i])
    return ret


def dev(G):
    """ Returns the deviator of G
    """
    return G - trace(G) / 3 * eye(3)
