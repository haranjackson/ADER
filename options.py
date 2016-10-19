n  = 5                      # Number of equations in the system of PDEs

""" Domain Parameters """

Lx = 1                      # Length of domain in x direction
Ly = 1                      # Length of domain in y direction
Lz = 1                      # Length of domain in z direction
nx = 100                    # Number of cells in x direction
ny = 20                     # Number of cells in y direction
nz = 1                      # Number of cells in z direction

""" General Solver Options """

N      = 2                  # Method is order N+1
para   = 0                  # Whether to parallelise the DG and FV calculations
ncore  = 4                  # Number of cores to use in parallelisation
method = 'rusanov'          # Method used for intercell fluxes ('osher' or 'rusanov')

""" WENO Parameters """

rc = 8                      # Exponent used in oscillation indicator
λc = 1e5                    # Coefficient of oscillation indicator of central stencil(s)
λs = 1                      # Coefficient of oscillation indicator of side stencils
ε  = 1e-14                  # Constant ensuring oscillation indicators don't blow up

""" DG Options """

hidalgo    = 0              # Whether to use the Hidalgo initial guess
stiff      = 0              # Whether source terms are stiff (Newton-Krylov solver is used)
superStiff = 0              # Whether to use Newton-Krylov to compute the Hidalgo initial guess
TOL        = 6e-6           # Tolerance to which the Galerkin Predictor must converge
MAX_ITER   = 50             # Maximum number of non-stiff iterations attempted in DG


""" Derived Values (do not change) """

from numpy import array
ndim = sum(array([nx, ny, nz]) > 1)
NT = (N+1)**(ndim+1)
dxi = [Lx/nx, Ly/ny, Lz/nz]             # Grid steps in the 3 spatial directions

method = method.lower()
if method not in ['rusanov', 'osher']:
    print('Warning: Flux type not recognised')
