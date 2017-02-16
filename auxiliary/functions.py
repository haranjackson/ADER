from numpy import concatenate, int64, kron, ones


def extend(inarray, extension, d):
    """ Extends the input array by the specified number of cells in the dth axis at both ends
    """
    n = inarray.shape[d]
    reps = concatenate(([extension+1], ones(n-2), [extension+1])).astype(int64)
    return inarray.repeat(reps, axis=d)

def kron_prod(matList):
    """ Returns the kronecker product of the matrices in matList
    """
    ret = matList[0]
    for i in range(1, len(matList)):
        ret = kron(ret, matList[i])
    return ret
