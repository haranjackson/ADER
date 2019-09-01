from numpy import array


def get_blocks(arr, N, ncore):
    """ Splits array of length n into ncore chunks. Returns the start and end
        indices of each chunk.
    """
    n = len(arr)
    step = int(n / ncore)
    inds = array([i * step for i in range(ncore)] + [n - N])
    inds[0] += N
    return [arr[inds[i] - N:inds[i + 1] + N] for i in range(len(inds) - 1)]
