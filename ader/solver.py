from itertools import product
from time import time
from concurrent.futures import ProcessPoolExecutor

from numpy import abs, array, amax, concatenate, zeros
from numpy.linalg import eigvals
from tangent import autodiff

from ader.etc.boundaries import standard_BC, periodic_BC
from ader.dg.dg import DGSolver
from ader.fv.fv import FVSolver
from ader.weno.weno import WENOSolver


def get_blocks(uBC, N, ncore):
    """ Splits array of length n into ncore chunks. Returns the start and end
        indices of each chunk.
    """
    n = len(uBC)
    step = int(n / ncore)
    inds = array([i * step for i in range(ncore)] + [n - N])
    inds[0] += N
    return [uBC[inds[i] - N: inds[i + 1] + N] for i in range(len(inds) - 1)]


def make_system_matrix(flux, nonconservative_matrix, NV):

    dFdQ = autodiff(flux)

    def system_matrix(Q, d, model_params=None):
        ret = zeros([NV, NV])
        for i in range(NV):
            x = zeros(NV)
            x[i] = 1
            ret[i] = dFdQ(Q, d, model_params, x)
        if nonconservative_matrix is not None:
            ret += nonconservative_matrix(Q, d, model_params)
        return ret

    return system_matrix


def make_max_eig(system_matrix):

    def max_eig(Q, d, model_params=None):
        M = system_matrix(Q, d, model_params)
        return amax(abs(eigvals(M)))

    return max_eig


class Solver():

    def __init__(self, nvar, ndim, flux, nonconservative_matrix=None, source=None,
                 system_matrix=None, max_eig=None, model_params=None,
                 order=2, riemann_solver='rusanov', ncore=1,
                 stiff_dg=False, stiff_dg_initial_guess=False,
                 newton_dg_initial_guess=False,
                 DG_TOL=6e-6, DG_IT=50,
                 WENO_r=8, WENO_λc=1e5, WENO_λs=1, WENO_ε=1e-14):

        self.NV = nvar
        self.NDIM = ndim
        self.N = order

        self.flux = flux
        self.nonconservative_matrix = nonconservative_matrix
        self.source = source
        self.model_params = model_params

        if system_matrix is None:
            self.system_matrix = make_system_matrix(flux,
                                                    nonconservative_matrix,
                                                    nvar)
        else:
            self.system_matrix = system_matrix

        if max_eig is None:
            self.max_eig = make_max_eig(self.system_matrix)
        else:
            self.max_eig = max_eig

        self.wenoSolver = WENOSolver(self.N, self.NV, self.NDIM,
                                     λc=WENO_λc, λs=WENO_λs, r=WENO_r,
                                     ε=WENO_ε)

        self.dgSolver = DGSolver(self.N, self.NV, self.NDIM,
                                 flux=flux, source=source,
                                 nonconservative_matrix=nonconservative_matrix,
                                 system_matrix=self.system_matrix,
                                 model_params=model_params,
                                 stiff=stiff_dg, stiff_ig=stiff_dg_initial_guess,
                                 nk_ig=newton_dg_initial_guess,
                                 tol=DG_TOL, max_iter=DG_IT)

        self.fvSolver = FVSolver(self.N, self.NV, self.NDIM,
                                 flux=flux, source=source,
                                 nonconservative_matrix=nonconservative_matrix,
                                 system_matrix=self.system_matrix,
                                 max_eig=self.max_eig,
                                 model_params=model_params,
                                 riemann_solver=riemann_solver)

        self.ncore = ncore

    def timestep(self, u, dX, count=None, t=None, final_time=None):
        """ Calculates dt, based on the maximum wavespeed across the domain
        """
        MAX = 0
        for coords in product(*[range(s) for s in u.shape[:self.NDIM]]):

            Q = u[coords]
            for d in range(self.NDIM):
                MAX = max(MAX, self.max_eig(Q, d, self.model_params) / dX[d])

        dt = self.cfl / MAX

        # Reduce early time steps to avoid initialization errors
        if count is not None and count <= 5:
            dt *= 0.2

        if final_time is not None:
            return min(final_time - t, dt)
        else:
            return dt

    def stepper(self, uBC, dt, dX, verbose=False):
        t0 = time()

        wh = self.wenoSolver.solve(uBC)
        t1 = time()

        qh = self.dgSolver.solve(wh, dt, dX)
        t2 = time()

        du = self.fvSolver.solve(qh, dt, dX)
        t3 = time()

        if verbose:
            print('WENO: {:.3f}s'.format(t1 - t0))
            print('DG:   {:.3f}s'.format(t2 - t1))
            print('FV:   {:.3f}s'.format(t3 - t2))
            print('Iteration Time: {:.3f}s\n'.format(time() - t0))

        return du

    def parallel_stepper(self, executor, uBC, dt, dX, verbose=False):
        t0 = time()

        blocks = get_blocks(uBC, self.N, self.ncore)
        n = len(blocks)
        chunks = executor.map(self.stepper, blocks, [dt] * n, [dX] * n)
        du = concatenate([c for c in chunks])

        if verbose:
            print('Iteration Time: {:.3f}s\n'.format(time() - t0))

        return du

    def resume(self, verbose=False):

        with ProcessPoolExecutor(max_workers=self.ncore) as executor:

            while self.t < self.final_time:

                dt = self.timestep(self.u, self.dX, count=self.count, t=self.t,
                                   final_time=self.final_time)

                if verbose:
                    print('Iteration:', self.count)
                    print('t  = {:.3e}'.format(self.t))
                    print('dt = {:.3e}'.format(dt))

                uBC = self.BC(self.u, self.N, self.NDIM)

                if self.ncore == 1:
                    self.u += self.stepper(uBC, dt, self.dX, verbose)
                else:
                    self.u += self.parallel_stepper(executor, uBC, dt, self.dX,
                                                    verbose)

                self.t += dt
                self.count += 1

                if self.callback is not None:
                    self.callback(self.u, self.t, self.count)

        return self.u

    def solve(self, initial_grid, final_time, dX, cfl=0.9,
              boundary_conditions='transitive', verbose=False, callback=None):

        self.u = initial_grid
        self.final_time = final_time
        self.dX = dX
        self.cfl = cfl
        self.callback = callback

        if boundary_conditions == 'transitive':
            self.BC = standard_BC
        elif boundary_conditions == 'periodic':
            self.BC = periodic_BC
        elif callable(boundary_conditions):
            self.BC = boundary_conditions
        else:
            raise ValueError("'boundary_conditions' must either be equal to " +
                             "'transitivie', 'periodic', or a callable function.")

        self.t = 0
        self.count = 0

        return self.resume(verbose)
