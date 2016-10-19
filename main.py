from data import u0, dt
from options import para
from system import boundary_condition
from auxiliary.parallel import para_predictor, para_fv_terms
from solver.weno import reconstruct
from solver.dg import predictor
from solver.fv import fv_terms


uBC = boundary_condition(u0)    # Apply boundary conditions
wh = reconstruct(uBC)           # Perform WENO reconstruction

if para:
    qh = para_predictor(wh, dt)           # Calculate DG predictor (in parallel)
    u1 = u0 + para_fv_terms(qh, dt)       # Update with FV terms (in parallel)
else:
    qh = predictor(wh, dt)                # Calculate DG predictor
    u1 = u0 + fv_terms(qh, dt)            # Update with FV terms
