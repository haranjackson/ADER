This is a Python implementation of the ADER-WENO method for solving hyperbolic systems of PDEs of the following form:

![Figure 1](/images/PDEsystem.png)

The non-conservative terms and the sources may of course be 0.

### Requirements

* Python 3
* NumPy
* SciPy
* Joblib (for parallelisation if desired, can be easily removed)

### Background

Given cell-wise constant initial data defined on a computational grid, this program performs an arbitrary-order polynomial reconstruction of the data in each cell, according to a modified version of the WENO method, as presented in [1].

To the same order, a spatio-temporal polynomial reconstruction of the data is then obtained in each spacetime cell by the Discontinuous Galerkin method, using the WENO reconstruction as initial data at the start of the timestep (see [2,3]).

Finally, a finite volume update step is taken, using the DG reconstruction to calculate the values of the intercell fluxes and non-conservative intercell jump terms, and the interior source terms and non-conservative terms (see [3]).

The intercell fluxes and non-conservative jumps are calculated using either a Rusanov-type flux [4], a Roe-type flux [5], or an Osher-type flux [6].

### Usage

The functions returning the flux terms, the non-conservative terms, and the source terms need to be specified in 'system.py' (along with the system Jacobian, if the Roe flux or the Osher flux is desired). The maximum of the absolute values of the eigenvalues should also be specified here (this can be found using SciPy's eigvals function applied to the system Jacobian if not known analytically).

The solver parameters and options are specified in 'options.py', including the order of the reconstruction, the size and shape of the grid, and the number of cores to parallelise over.

'main.py' contains an example implementation using the Godunov-Peshkov-Romenski model with an Ideal Gas equation of state. The test case consists of a 1D viscous shock travelling from left to right at Mach 2.

### NOTE

This implementation is pretty slow. It is intended to be used only for academic purposes. If you have a commercial application that requires a rapid, bullet-proof implementation of the ADER-WENO method or the GPR model, then get in touch (jackson.haran@gmail.com).

### References

1. Dumbser, Zanotti, Hidalgo, Balsara - *ADER-WENO finite volume schemes with space-time adaptive mesh refinement*
2. Dumbser, Castro, Pares, Toro - *ADER schemes on unstructured meshes for nonconservative hyperbolic systems: Applications to geophysical flows*
3. Dumbser, Hidalgo, Zanotti - *High order space-time adaptive ADER-WENO finite volume schemes for non-conservative hyperbolic systems*
4. Toro - *Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction*
5. Dumbser, Toro - *On Universal Osher-Type Schemes for General NonlinearHyperbolic Conservation Laws*
6. Dumbser, Toro - *A simple extension of the Osher Riemann solver to non-conservative hyperbolic systems*
