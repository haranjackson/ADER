from itertools import product
from time import time
from concurrent.futures import ProcessPoolExecutor

from numpy import array, concatenate, expand_dims

from ader.etc.autodiff import make_M, make_max_eig
from ader.etc.boundaries import standard_BC, periodic_BC, extend_mask, \
    extend_grid
from ader.dg.dg import DGSolver
from ader.fv.fv import FVSolver
from ader.weno.weno import WENOSolver


def get_blocks(arr, N, ncore):
    """ Splits array of length n into ncore chunks. Returns the start and end
        indices of each chunk.
    """
    n = len(arr)
    step = int(n / ncore)
    inds = array([i * step for i in range(ncore)] + [n - N])
    inds[0] += N
    return [arr[inds[i] - N: inds[i + 1] + N] for i in range(len(inds) - 1)]


class Solver():
    """ Class for solving the following system of PDEs using the ADER method

    .. math::

        \\frac{\partial Q}{\partial t} + \\nabla \cdot F(Q) + B(Q) \cdot  \\nabla Q = S(Q)

    which can also be written as:

    .. math::

        \\frac{\partial Q}{\partial t} + M(Q) \cdot  \\nabla Q = S(Q)

    where

    .. math::

         M(Q) = \\frac{\partial F}{\partial Q} + B(Q)

    Required Parameters
    -------------------
    nvar : int
        The number of conserved variables (the length of vector Q, equal to the
        number of equations in the system)
    ndim : int
        The number of spatial dimensions
    F : function
        Must have signature `(Q, d, model_params)` and return NumPy array of
        size `nvar`. Returns :math:`F(Q)` in direction d. See model_params below.
    B : function (default None)
        Must have signature `(Q, d, model_params)` and return NumPy array of
        size `nvar×nvar`. Returns :math:`B(Q)` in direction d. See model_params below.
        Not necessary if system is conservative.
    S : function (default None)
        Must have signature `(Q, model_params)` and return NumPy array of size
        `nvar`. Returns :math:`S(Q)`. See model_params below. Not necessary if system
        is homogeneous.
    model_params : object (default None)
        An object containing any additional parameters required by the model.
        Not necessary if no additional parameters are required.

    Recommended Parameters
    ----------------------
    M : function (default None)
        Must have signature `(Q, d, model_params)` and return NumPy array of
        size `nvar×nvar`. Returns M(Q) in direction d. See model_params below.
        If None, M(Q) is calculated by automatic differentiation.
    max_eig : function (default None)
        Must have signature `(Q, d, model_params)`. Returns eigenvalue of
        maximum absolute value of M(Q) in direction d. See model_params below.
        If None, max_eig is calculated from M(Q) by automatic differentiation.
    order : int (default 2)
        The order of accuracy to which the solution should be found.
    ncore : int (default 1)
        The number of CPU cores to be used in calculating the solution.

    Advanced Parameters
    -------------------
    riemann_solver : string (default 'rusanov')
        Which Riemann solver to use. Options: 'rusanov', 'roe', 'osher'.
    stiff_dg : bool (default False)
        Whether to use a Newton Method to solve the root finding involved in
        calculating the DG predictor.
    stiff_dg_guess : bool (default False)
        Whether to use an advanced initial guess for the DG predictor (only for
        very stiff systems).
    newton_dg_guess : bool (default False)
        Whether to compute the advanced initial guess using a Newton Method
        (only for very, very stiff systems).
    DG_TOL : float (default 6e-6)
        The tolerance to which the DG predictor is calculated.
    DG_IT : int (default 50)
        Maximum number of iterations attempted if solving the DG root finding
        problem iteratively (if not using a Newton Method).
    WENO_r : float (default 8)
        The WENO exponent r.
    WENO_λc : float (default 1e5)
        The WENO weighting of the central stencils.
    WENO_λs : float (default 1)
        The WENO weighting of the side stencils.
    WENO_ε : float (default 1e-14)
        The constant used in the WENO method to avoid numerical issues.

    Returns
    -------
    Solver : class
        Class with method `solve`, used to solve the equations for a particular
        initial grid and final time.

    Notes
    -----
    Given cell-wise constant initial data defined on a computational grid,
    this program performs an arbitrary-order polynomial reconstruction of
    the data in each cell, according to a modified version of the WENO
    method, as presented in [1].

    To the same order, a spatio-temporal polynomial reconstruction of the
    data is then obtained in each spacetime cell by the Discontinuous
    Galerkin method, using the WENO reconstruction as initial data at the
    start of the timestep (see [2,3]).

    Finally, a finite volume update step is taken, using the DG
    reconstruction to calculate the values of the intercell fluxes and
    non-conservative intercell jump terms, and the interior source terms and
    non-conservative terms (see [3]).

    The intercell fluxes and non-conservative jumps are calculated using
    either a Rusanov-type flux [4], a Roe-type flux [5], or an Osher-type
    flux [6].

    References
    ----------

    1. Dumbser, Zanotti, Hidalgo, Balsara - *ADER-WENO finite volume schemes
       with space-time adaptive mesh refinement*
    2. Dumbser, Castro, Pares, Toro - *ADER schemes on unstructured meshes
       for nonconservative hyperbolic systems: Applications to geophysical
       flows*
    3. Dumbser, Hidalgo, Zanotti - *High order space-time adaptive ADER-WENO
       finite volume schemes for non-conservative hyperbolic systems*
    4. Toro - *Riemann Solvers and Numerical Methods for Fluid Dynamics: A
       Practical Introduction*
    5. Dumbser, Toro - *On Universal Osher-Type Schemes for General
       Nonlinear Hyperbolic Conservation Laws*
    6. Dumbser, Toro - *A simple extension of the Osher Riemann solver to
       non-conservative hyperbolic systems*
    """
    def __init__(self, nvar, ndim, F, B=None, S=None, model_params=None,
                 M=None, max_eig=None, order=2, ncore=1,
                 riemann_solver='rusanov', stiff_dg=False,
                 stiff_dg_guess=False, newton_dg_guess=False,
                 DG_TOL=6e-6, DG_MAX_ITER=50, WENO_r=8, WENO_λc=1e5, WENO_λs=1,
                 WENO_ε=1e-14):

        self.NV = nvar
        self.NDIM = ndim
        self.N = order

        self.F = F
        self.B = B
        self.S = S
        self.pars = model_params

        if M is None:
            self.M = make_M(self.F, self.B, self.NV)
        else:
            self.M = M

        if max_eig is None:
            self.max_eig = make_max_eig(self.M)
        else:
            self.max_eig = max_eig

        self.wenoSolver = WENOSolver(self.N, self.NV, self.NDIM, λc=WENO_λc,
                                     λs=WENO_λs, r=WENO_r, ε=WENO_ε)

        self.dgSolver = DGSolver(self.N, self.NV, self.NDIM, F=self.F,
                                 S=self.S, B=self.B, M=self.M, pars=self.pars,
                                 stiff=stiff_dg, stiff_guess=stiff_dg_guess,
                                 newton_guess=newton_dg_guess, tol=DG_TOL,
                                 max_iter=DG_MAX_ITER)

        self.fvSolver = FVSolver(self.N, self.NV, self.NDIM, F=self.F,
                                 S=self.S, B=self.B, M=self.M,
                                 max_eig=self.max_eig, pars=self.pars,
                                 riemann_solver=riemann_solver)

        self.ncore = ncore

    def timestep(self, mask=None):
        """ Calculates dt, based on the maximum wavespeed across the domain
        """
        MAX = 0
        for coords in product(*[range(s) for s in self.u.shape[:self.NDIM]]):

            if mask is None or mask[coords]:
                Q = self.u[coords]
                for d in range(self.NDIM):
                    MAX = max(MAX, self.max_eig(Q, d, self.pars) / self.dX[d])

        dt = self.cfl / MAX

        # Reduce early time steps to avoid initialization errors
        if self.count <= 5:
            dt /= 5

        return min(self.final_time - self.t, dt)

    def ader_stepper(self, uBC, dt, verbose, maskBC):
        """ Calculates the change in the grid over time step dt.
        """
        t0 = time()

        wh = self.wenoSolver.solve(uBC)
        t1 = time()

        qh = self.dgSolver.solve(wh, dt, self.dX, maskBC)
        t2 = time()

        du = self.fvSolver.solve(qh, dt, self.dX, maskBC)
        t3 = time()

        if verbose:
            print('WENO: {:.3f}s'.format(t1 - t0))
            print('DG:   {:.3f}s'.format(t2 - t1))
            print('FV:   {:.3f}s'.format(t3 - t2))
            print('Iteration Time: {:.3f}s\n'.format(time() - t0))

        return du

    def parallel_ader_stepper(self, executor, uBC, dt, maskBC):
        """ Splits uBC into equally-spaced chunks in the x-axis, and applies
            the ADER stepper to each.
        """
        t0 = time()

        blocks = get_blocks(uBC, self.N, self.ncore)
        n = len(blocks)

        if maskBC is None:
            maskBlocks = [None] * n
        else:
            maskBC_ = extend_grid(maskBC, self.N - 1, 0, 0)
            maskBlocks = [mb[self.N - 1 : -(self.N - 1)]
                          for mb in get_blocks(maskBC_, self.N, self.ncore)]

        chunks = executor.map(self.ader_stepper, blocks, [dt] * n, [False] * n,
                              maskBlocks)
        du = concatenate([c for c in chunks])

        if self.verbose:
            print('Iteration Time: {:.3f}s\n'.format(time() - t0))

        return du

    def stepper(self, executor, dt, mask=None):
        """ Steps the grid forward in time by dt. If mask is not None, then
            only the cells (i,j,...) where mask(i,j,...)=True are updated.
        """
        uBC = self.BC(self.u, self.N, self.NDIM)

        if mask is None:
            maskBC = None
        else:
            maskBC = extend_mask(mask)

        if self.ncore == 1:
            du = self.ader_stepper(uBC, dt, self.verbose, maskBC)
        else:
            du = self.parallel_ader_stepper(executor, uBC, dt, maskBC)

        if mask is None:
            self.u += du
        else:
            self.u += du * expand_dims(mask, -1)

    def resume(self):

        with ProcessPoolExecutor(max_workers=self.ncore) as executor:

            while self.t < self.final_time:

                dt = self.timestep()

                if self.verbose:
                    print('Iteration:', self.count)
                    print('t  = {:.3e}'.format(self.t))
                    print('dt = {:.3e}'.format(dt))

                self.stepper(executor, dt)

                self.t += dt
                self.count += 1

                if self.callback is not None:
                    self.callback(self.u, self.t, self.count)

        return self.u

    def solve(self, initial_grid, final_time, dX, cfl=0.9,
              boundary_conditions='transitive', verbose=False, callback=None):
        """ Solves the system of PDEs, given the initial grid and final time.

        Parameters
        ----------
        initial_grid : array
            A NumPy array of ndim + 1 dimensions. Cell
            :math:`(i_1, i_2, ..., i_{ndim}, j)` corresponds to the
            jth variable in cell :math:`(i_1, i_2, ..., i_{ndim})` of the
            domain.
        final_time : float
            The final time to which the solution should run
        dX : list / array
            A list of length ndim, containing the grid spacing in each spatial
            axis
        cfl : float (default 0.9)
            The CFL number (must be < 1)
        boundary_conditions : string / function (default 'transitive')
            Which kind of boundary conditions to use. Options: 'transitive',
            'periodic', function with signature (grid, N, ndim). In the latter
            case, the function should return a NumPy array with the same number
            of axes as grid, but with N more cells at each end of the grid in
            each spatial direction. These extra cells are required by an
            N-order method.
        verbose : bool (default False)
            Whether to print full output at each timestep.
        callback : function (default None)
            A user-defined callback function with signature (grid, t, count)
            where grid is the value of the computational grid at time t (and
            timestep count).

        Returns
        -------
        result : array
            The value of the grid at time t=final_time.
        """
        self.u = initial_grid
        self.final_time = final_time
        self.dX = dX
        self.cfl = cfl
        self.verbose = verbose
        self.callback = callback

        if boundary_conditions == 'transitive':
            self.BC = standard_BC
        elif boundary_conditions == 'periodic':
            self.BC = periodic_BC
        elif callable(boundary_conditions):
            self.BC = boundary_conditions
        else:
            raise ValueError("'boundary_conditions' must either be equal to " +
                             "'transitivie', 'periodic', or a function.")

        self.t = 0
        self.count = 0

        return self.resume()
