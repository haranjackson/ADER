from numpy import concatenate, int64, ones


def extend(inarray, extension, d):
    """ Extends the input array by M cells on each surface
    """
    n = inarray.shape[d]
    reps = concatenate(([extension + 1], ones(n - 2),
                        [extension + 1])).astype(int64)
    return inarray.repeat(reps, axis=d)
