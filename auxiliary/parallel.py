from joblib import delayed, Parallel
from numpy import array, concatenate

from options import ncore
from solver.dg import predictor
from solver.fv import fv_terms


def para_predictor(wh, dt):
    """ Controls the parallel computation of the Galerkin predictor
    """
    pool = Parallel(n_jobs=ncore)
    nx = wh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    m = len(chunk) - 1
    qhList = pool(delayed(predictor)(wh[chunk[i]:chunk[i+1]], dt) for i in range(m))
    return concatenate(qhList)

def para_fv_terms(qh, dt):
    """ Controls the parallel computation of the Finite Volume interface terms
    """
    pool = Parallel(n_jobs=ncore)
    nx = qh.shape[0]
    step = int(nx / ncore)
    chunk = array([i*step for i in range(ncore)] + [nx+1])
    chunk[0] += 1
    chunk[-1] -= 1
    m = len(chunk) - 1
    fvList = pool(delayed(fv_terms)(qh[chunk[i]-1:chunk[i+1]+1], dt) for i in range(m))
    return concatenate(fvList)
