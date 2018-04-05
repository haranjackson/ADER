ADER
====

A Python implementation of the ADER method for solving any (potentially
very stiff) hyperbolic system of PDEs of the following form:

.. figure:: http://quicklatex.com/cache3/d6/ql_47fb6a15d79d64bde7b461d1ff5346d6_l3.png
   :alt: Figure 1

   Figure 1

An arbitrary number of spatial domains can be used, and this
implementation is capable of solving the equations to any order of
accuracy. Second-order parabolic PDEs will be implemented soon.

Installation
------------

Run ``pip install ader``

The following dependencies are required:

-  Python 3.2+
-  NumPy 1.13+
-  SciPy 0.19+
-  Tangent 0.1.9+

Background
----------

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

Usage
-----

Defining the System of Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user must define the following function, corresponding to F(Q):

``flux(Q, d, model_params)``

Given a vector of conserved variables ``Q``, ``flux`` must return the
flux in direction ``d`` (where ``d=0,1,...`` corresponds to the
x-,y-,... axes) as a NumPy array. ``model_params`` should be a dict
object containing any other parameters required by the model in the
calculation of the flux (for example, the heat capacity ratio in the
Euler equations, or the viscosity in the Navier-Stokes equations).
``model_params`` does not have to be used, but it must be contained in
the signature of ``flux``.

Similarly, if they are required, the nonconservative matrix B(Q) in
direction d must be defined in a function with the following signature
(as a square NumPy array):

``nonconservative_matrix(Q, d, model_params)``

and the source terms must be defined in a function with the following
signature (as a NumPy array):

``source(Q, model_params)``

If analytical forms are available, the system Jacobian in direction d
should be defined in a function with the following signature:

``system_matrix(Q, d, model_params)``

and the eigenvalue of the system in direction d with largest absolute
value should be defined in a function with the following signature:

``max_eig(Q, d, model_params)``

The system Jacobian and largest absolute eigenvalue are required for the
ADER method. If no analytical forms are available, they will be derived
from the supplied flux (and nonconservative matrix, if supplied) using
`automatic
differentiation <https://en.wikipedia.org/wiki/Automatic_differentiation>`__.
This will introduce some computational overhead, however, and some
considerations need to be made when defining the flux function (see the
dedicated section below).

Solving the Equations
~~~~~~~~~~~~~~~~~~~~~

Suppose we are solving the `reactive Euler
equations <https://www.sciencedirect.com/science/article/pii/0895717796001471>`__
in 2D. We define ``flux`` and ``source`` as above (but not
``nonconservative_matrix``, as the equations are conservative). We also
define ``model_params`` to hold the heat capacity ratio and reaction
constants. These equations contain 5 conserved variables. To solve them
to order 3, with 4 CPU cores, we set up the solver object thus:

::

    from ader.solver import Solver

    solver = Solver(nvar=5, ndim=2, flux=flux, nonconservative_matrix=None, source=source, model_params=model_params, order=3, ncore=4)

Analytical forms of the system Jacobian and the eigenvalue of largest
absolute size exist for the reactive Euler equations. If we define these
in ``system_matrix`` and ``max_eig`` we may instead create the solver
object thus:

::

    solver = Solver(nvar=5, ndim=2, flux=flux, nonconservative_matrix=None, source=source, system_matrix=system_matrix, max_eig=max_eig, model_params=model_params, order=3, ncore=4)

To solve a particular problem, we must define the initial state,
``initial_grid``. This must be a NumPy array with 3 axes, such that
``initial_grid[i,j,k]`` is equal to the value of the kth conserved
variable in cell (i,j). We must also define list ``dX=[dx,dy]`` where
dx,dy are the grid spacing in the x and y axes. To solve the problem to
a final time of 0.5, with a CFL number of 0.9, while printing all
output, we call:

::

    solution = solver.solve(initial_grid, 0.5, dX, cfl=0.9, verbose=True)

Advanced Usage
~~~~~~~~~~~~~~

The Solver class has the following additional arguments:

-  **riemann\_solver** (default 'rusanov'): Which Riemann solver should
   be used. Options: 'rusanov', 'roe', 'osher'.
-  **stiff\_dg** (default False): Whether to use a Newton Method to
   solve the root finding involved in calculating the DG predictor.
-  **stiff\_dg\_initial\_guess** (default False): Whether to use an
   advanced initial guess for the DG predictor (only for very stiff
   systems).
-  **newton\_dg\_initial\_guess** (default False): Whether to compute
   the advanced initial guess using a Newton Method (only for very, very
   stiff systems).
-  **DG\_TOL** (default 6e-6): The tolerance to which the DG predictor
   is calculated.
-  **DG\_IT** (default 50): Maximum number of iterations attempted if
   solving the DG root finding problem iteratively (not with a Newton
   Method)
-  **WENO\_r** (default 8): The WENO exponent r.
-  **WENO\_λc** (default 1e5): The WENO weighting of the central
   stencils.
-  **WENO\_λs** (default 1): The WENO weighting of the side stencils.
-  **WENO\_ε** (default 1e-14): The constant used in the WENO method to
   avoid numerical issues.

The Solver.solve method has the following additional arguments:

-  **boundary\_conditions** (default 'transitive'): Which kind of
   boundary conditions to use. Options: 'transitive', 'periodic',
   ``func(grid, N, ndim)``. In the latter case, the user defines a
   function with the stated signature. It should return a NumPy array
   with the same number of axes as grid, but with ``N`` more cells on
   either side of the grid in each spatial direction. These extra cells
   are required by an N-order method.
-  **callback** (default None): A user-defined callback function with
   signature ``callback(grid, t, count)`` where ``grid`` is the value of
   the computational grid at time ``t`` (and timestep ``count``).

Examples
~~~~~~~~

Check out example.py to see a couple of problems being solved for the
GPR model and the reaction Euler equations.

Notes
-----

Speed
~~~~~

This implementation is pretty slow. It's really only intended to be used
only for academic purposes. If you have a commercial application that
requires a rapid, bullet-proof implementation of the ADER method or the
GPR model, then get in touch (jackson.haran@gmail.com).

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

The automatic differentiation used if ``system_matrix`` and ``max_eig``
is performed using `Google's Tangent
library <https://github.com/google/tangent>`__. Although it's great,
this library is quite new, and it cannot cope with all operations that
you may use in your fluxes (although development is proceeding quickly).
In particular, it will never be able to handle closures, and classes are
not yet implemented. Some NumPy functions such as ``inv`` have not yet
been implemented. If you run into issues, drop me a quick message and
I'll let you know if I can make it work.

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
