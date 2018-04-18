from itertools import product

from numpy import array, dot, tensordot, zeros

from ader.etc.basis import Basis, derivative
from ader.fv.fluxes import B_INT, D_OSH, D_RUS, D_ROE
from ader.fv.matrices import quad_weights


def endpoints(qh, NDIM, ENDVALS):
    """ Returns tensor T where T[d,e,i1,...,in] is the set of DG coefficients
        in the dth direction, at end e (either 0 or 1), in cell (i1,...,in)
    """
    return array([tensordot(ENDVALS, qh, (0, NDIM + 1 + d))
                  for d in range(NDIM)])


class FVSolver():

    def __init__(self, N, NV, NDIM, F, S=None, B=None, M=None, max_eig=None,
                 pars=None, riemann_solver='rusanov', time_rec=True):

        self.N = N
        self.NV = NV
        self.NDIM = NDIM

        self.F = F
        self.S = S
        self.B = B
        self.M = M
        self.max_eig = max_eig
        self.pars = pars

        if riemann_solver == 'rusanov':
            self.D_FUN = D_RUS
        elif riemann_solver == 'roe':
            self.D_FUN = D_ROE
        elif riemann_solver == 'osher':
            self.D_FUN = D_OSH
        else:
            raise ValueError("Choice of 'riemann_solver' not recognised.\n" +
                             "Choose from 'rusanov', 'roe', and 'osher'.")

        self.time_rec = time_rec

        basis = Basis(N)
        self.NODES = basis.NODES
        self.WGHTS = basis.WGHTS
        self.ENDVALS = basis.ENDVALS
        self.DERVALS = basis.DERVALS
        self.TN, self.WGHT, self.WGHT_END = quad_weights(N, NDIM, basis.WGHTS,
                                                         time_rec)

    def interfaces(self, ret, qEnd, dX, mask):
        """ Returns the contribution to the finite volume update coming from
            the fluxes at the interfaces
        """
        dims = ret.shape[:self.NDIM]
        nweights = len(self.WGHT_END)

        for d in range(self.NDIM):

            # dimensions of cells used when calculating fluxes in direction d
            coordList = [range(1, dim + 1) for dim in dims[:d]] + \
                        [range(1, dims[d] + 2)] + \
                        [range(1, dim + 1) for dim in dims[d + 1:]]

            for rcoords in product(*coordList):

                lcoords = rcoords[:d] + (rcoords[d] - 1,) + rcoords[d + 1:]

                if mask is None or (mask[lcoords] and mask[rcoords]):

                    # qL,qR are the sets of polynomial coefficients for the DG
                    # reconstruction at the left & right sides of the interface
                    qL = qEnd[(d, 1) + lcoords].reshape(nweights, self.NV)
                    qR = qEnd[(d, 0) + rcoords].reshape(nweights, self.NV)

                    # integrate the flux over the surface normal to direction d
                    fInt = zeros(self.NV)    # flux from conservative terms
                    BInt = zeros(self.NV)    # flux from non-conservative terms
                    for ind in range(nweights):
                        qL_ = qL[ind]
                        qR_ = qR[ind]

                        fInt += self.WGHT_END[ind] * self.D_FUN(self,
                                                                qL_, qR_, d)

                        if self.B is not None:
                            BInt += self.WGHT_END[ind] * B_INT(self,
                                                               qL_, qR_, d)

                    rcoords_ = tuple(c - 1 for c in rcoords)
                    lcoords_ = rcoords_[:d] + (rcoords_[d] - 1,) + rcoords_[d + 1:]

                    if lcoords_[d] >= 0:
                        ret[lcoords_] -= (fInt + BInt) / dX[d]

                    if rcoords_[d] < dims[d]:
                        ret[rcoords_] += (fInt - BInt) / dX[d]

    def centers(self, ret, qh, dX, mask):
        """ Returns the space-time averaged source term and non-conservative
            terms
        """
        for coords in product(*[range(dim) for dim in ret.shape[:self.NDIM]]):

            coords_ = tuple(coord + 1 for coord in coords)

            if mask is None or mask[coords_]:

                qhi = qh[coords_]

                # Integrate across volume of spacetime cell
                for inds in product(*[range(s) for s in self.WGHT.shape]):

                    q = qhi[inds]
                    tmp = zeros(self.NV)

                    if self.S is not None:
                        tmp = self.S(q, self.pars)

                    if self.B is not None:

                        qt = qhi[inds[0]]

                        for d in range(self.NDIM):

                            dqdx = derivative(self.N, self.NV, self.NDIM, qt,
                                              inds[1:], d, self.DERVALS)

                            B = self.B(q, d, self.pars)
                            Bdqdx = dot(B, dqdx)

                            tmp -= Bdqdx / dX[d]

                    ret[coords] += self.WGHT[inds] * tmp

    def solve(self, qh, dt, dX, mask=None):
        """ Returns the space-time averaged interface terms, jump terms,
            source terms, and non-conservative terms
        """
        if not self.time_rec:
            qh = qh.reshape(qh.shape[:self.NDIM] + (1,) + qh.shape[self.NDIM:])

        qEnd = endpoints(qh, self.NDIM, self.ENDVALS)

        ret = zeros([s - 2 for s in qh.shape[:self.NDIM]] + [self.NV])

        if self.S is not None or self.B is not None:
            self.centers(ret, qh, dX, mask)

        self.interfaces(ret, qEnd, dX, mask)

        return dt * ret
