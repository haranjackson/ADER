""" Domain Parameters """

nV = 17                     # Number of variables in the equations
Lx = 1                      # Length of domain in x direction
Ly = 1                      # Length of domain in x direction
Lz = 1                      # Length of domain in x direction
nx = 100                    # Number of cells in x direction
ny = 1                      # Number of cells in y direction
nz = 1                      # Number of cells in z direction

""" General Solver Options """

N = 2                       # Order of the method
CFL = 0.9                   # CFL number
FLUX = 0                    # Flux type (0=Rusanov, 1=Roe, 2=Osher)

""" DG Options """

STIFF = True                # Use Newton-Krylov to solve the DG system
DG_TOL = 1e-6               # Tolerance to which the predictor must converge
DG_IT = 50                  # Max number of non-stiff iterations attempted

""" WENO Parameters """

rc = 8                      # Exponent used in oscillation indicator
λc = 1e5                    # WENO coefficient of central stencils
λs = 1                      # WENO coefficient of side stencils
ε = 1e-14                   # Ensures oscillation indicators don't blow up

""" Speed-Up Options """

PARA_DG = False             # Parallelise DG step
PARA_FV = False             # Parallelise FV step
NCORE = 4                   # Number of cores used if running in parallel


""" Derived Values (do not change) """

ndim = (nx > 1) + (ny > 1) + (nz > 1)
NT = N**(ndim + 1)
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz
