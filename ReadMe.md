This is a Python implementation of the ADER-WENO method for solving hyperbolic systems of PDEs of the following form:

![Figure 1](/images/PDEsystem.png)

The non-conservative terms and the sources may of course be 0.

### Requirements

* Python 3
* NumPy
* SciPy
* Numba (for just-in-time compilation of performance-critical bits of code, can be easily removed)
* Joblib (for parallelisation if desired, can be easily removed)

### Background

Given cell-wise constant initial data defined on a computational grid, this program performs an arbitrary-order polynomial reconstruction of the data in each cell, according to a modified version of the WENO method, as presented in [1].

To the same order, a spatio-temporal polynomial reconstruction of the data is then obtained in each spacetime cell by the Discontinuous Galerkin method, using the WENO reconstruction as initial data at the start of the timestep (see [2,3]).

The DG method involves finding the root of a nonlinear system. A slight modification of the initial guess proposed in [4] for systems with stiff source terms is implemented here. An option to use the WENO reconstruction as the initial guess is also available, as in [5].

Finally, a finite volume update step is taken, using the DG reconstruction to calculate the values of the intercell fluxes and non-conservative intercell jump terms, and the interior source terms and non-conservative terms (see [3]).

The intercell fluxes and non-conservative jumps are calculated using either a Rusanov-type flux [6] or an Osher-Solomon-type flux [7].

### Usage

The functions returning the flux terms, the non-conservative terms, and the source terms are specified in 'system.py', along with the Jacobian of the system if Hidalgo's initial guess is being used for the root of the DG system. The maximum of the absolute values of the eigenvalues should also be specified here if the Rusanov-type intercell flux is being used in the FV update, but this can be found using SciPy's eigvals function if not known analytically. The boundary conditions can also be specified in this file.

The solver parameters and options are specified in 'options.py', including the order of the reconstruction, the size and shape of the grid, and the number of cores to parallelise over.

The initial data and timestep are provided in 'data.py'.

'main.py' contains a short script to demonstrate the process over one timestep.

### References

1. Dumbser, Zanotti, Hidalgo, Balsara - *ADER-WENO finite volume schemes with space-time adaptive mesh refinement*
2. Dumbser, Castro, Pares, Toro - *ADER schemes on unstructured meshes for nonconservative hyperbolic systems: Applications to geophysical flows*
3. Dumbser, Hidalgo, Zanotti - *High order space-time adaptive ADER-WENO finite volume schemes for non-conservative hyperbolic systems*
4. Hidalgo, Dumbser - *ADER schemes for nonlinear systems of stiff advection-diffusion-reaction equations*
5. Dumbser, Hidalgo, Castro, Pares, Toro - *FORCE schemes on unstructured meshes II: Non-conservative hyperbolic systems*
6. Toro - *Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction*
7. Dumbser, Toro - *A simple extension of the Osher Riemann solver to non-conservative hyperbolic systems*
