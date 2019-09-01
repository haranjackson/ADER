from itertools import product

from numpy import arange, array, tensordot, zeros


def endpoints(qh, NDIM, ENDVALS):
    """ Returns tensor T where T[d,e,i1,...,in] is the set of DG coefficients
        in the dth direction, at end e (either 0 or 1), in cell (i1,...,in)
    """
    return array(
        [tensordot(ENDVALS, qh, (0, NDIM + 1 + d)) for d in range(NDIM)])


def quad_weights(N, NDIM, WGHTS, time_rec=True):
    """ WGHT contains quadrature weights for integration over a spacetime cell
        (the cartesian product of the temporal and spatial quadrature weights)

        WGHT_END contains quadrature weights for integration over the
        spacetime hypersurface normal to a particular spatial direction.
        Th factor of 0.5 comes from the factor of 1/2 in fluxes - applied here
        for numerical reasona
    """
    TWGHTS = WGHTS if time_rec else array([1])
    TN = len(TWGHTS)

    WGHT = zeros([TN] + [N] * NDIM)
    indList = [arange(N)] * NDIM

    for t in range(TN):
        for inds in product(*indList):

            wght = TWGHTS[t]
            for d in range(NDIM):
                wght *= WGHTS[inds[d]]

            WGHT[(t,) + inds] = wght

    WGHT_END = 0.5 * WGHT.sum(axis=-1).ravel()

    return TN, WGHT, WGHT_END
