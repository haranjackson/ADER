from numpy import dot, sum, zeros
from numpy.linalg import det


def gram(A):
    """ Returns the gramian matrix of A
    """
    return dot(A.T, A)


def I_1(G):
    """ Returns the first invariant of G
    """
    return G.trace()


def I_2(G):
    """ Returns the second invariant of G
    """
    return 1 / 2 * (G.trace()**2 - sum(G * G))


def I_3(G):
    """ Returns the third invariant of G
    """
    return det(G)


def inv3(m):
    """ Inverse of 3x3 matrix m (required by tangent libary)
    """
    det = m[0, 0] * (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) - \
             m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) + \
             m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])

    minv = zeros([3, 3])
    minv[0, 0] = (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) / det
    minv[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) / det
    minv[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) / det
    minv[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) / det
    minv[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) / det
    minv[1, 2] = (m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2]) / det
    minv[2, 0] = (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]) / det
    minv[2, 1] = (m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1]) / det
    minv[2, 2] = (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) / det

    return minv