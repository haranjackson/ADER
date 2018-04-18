from itertools import product

from numpy import arange, array, concatenate, flip, int64, ones, prod, zeros


def extend_grid(arr, ext, d, kind):
    """ Extends the arr by ext cells on each surface in direction d.
        kind=0: transmissive, kind=1: no-slip, kind=2: periodic
    """
    n = arr.shape[d]

    if kind == 0:
        reps = concatenate((zeros(ext),
                            arange(n),
                            (n - 1) * ones(ext))).astype(int64)
    elif kind == 1:
        reps = concatenate((flip(arange(ext), 0),
                            arange(n),
                            flip(arange(n - ext, n), 0))).astype(int64)
    elif kind == 2:
        reps = concatenate((arange(n - ext, n),
                            arange(n),
                            arange(ext))).astype(int64)

    return arr.take(reps, axis=d)


def standard_BC(u, N, NDIM, wall=None, reflectVars=None):
    """ Extends the grid u in all dimensions. If wall[d]=1 then the
        boundaries in dimension d are no-slip, else they are transmissive.
    """
    if wall is None:
        wall = [False] * NDIM

    ret = u.copy()
    endCells = concatenate([arange(N), arange(-N, 0)])

    for d in range(NDIM):

        wall_ = wall[d] != 0
        ret = extend_grid(ret, N, d, wall_)

        if wall_:
            shape = ret.shape
            n1 = prod(shape[:d])
            n2 = shape[d]
            n3 = prod(shape[d + 1: NDIM])
            ret.reshape(n1, n2, n3, -1)[:, endCells, :, reflectVars] *= -1

    return ret


def periodic_BC(u, N, NDIM):
    """ Applies period boundary conditions to u, of length N on each side
    """
    ret = u.copy()
    for d in range(NDIM):
        ret = extend_grid(ret, N, d, 2)
    return ret


def neighbor_cells(arr, coords):
    """ Returns the neighbors of the cell in arr given by coords
    """
    ndim = arr.ndim
    shape = arr.shape
    return [arr[coords[:d] + (coords[d] + 1,) + coords[d + 1:]]
            for d in range(ndim) if coords[d] + 1 < shape[d]] + \
           [arr[coords[:d] + (coords[d] - 1,) + coords[d + 1:]]
            for d in range(ndim) if coords[d] > 0]


def extend_mask(mask):
    """ Given a mask corresponding to the cells that the user wishes to update,
        this function returns a mask of the cells for which the DG predictor
        must be calculated (i.e. the masked cells and their neighbors)
    """
    mask = array(mask, dtype=int)

    for coords in product(*[range(s) for s in mask.shape]):

        if mask[coords] == 0:

            neighbors = neighbor_cells(mask, coords)

            if 1 in neighbors:
                mask[coords] = 2

    ret = array(mask, dtype=bool)
    for d in range(mask.ndim):
        ret = extend_grid(ret, 1, d, 0)

    return ret
