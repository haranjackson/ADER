from numpy import zeros

from example.gpr.systems.conserved import flux_cons_ref, Bdot_cons
from example.gpr.systems.conserved import block_cons_ref, source_cons_ref
from example.gpr.systems.eigenvalues import max_abs_eigs
from options import nV


def flux_ref(ret, Q, d):
    """ ret is modified in-place, to consist of the flux F, in direction d,
        given state Q
    """
    flux_cons_ref(ret, Q, d)

def block_ref(ret, Q, d):
    """ ret is modified in-place, to consist of the non-conservative matrix B,
        in direction d, given state Q
    """
    block_cons_ref(ret, Q, d)

def source_ref(ret, Q):
    """ ret is modified in-place, to consist of the source term S,
        given state Q
    """
    source_cons_ref(ret, Q)

def flux(Q, d):
    """ returns the flux F, in direction d, given state Q
    """
    ret = zeros(nV)
    flux_ref(ret, Q, d)
    return ret

def block(Q, d):
    """ returns the non-conservative matrix B, in direction d, given state Q
    """
    ret = zeros([nV, nV])
    block_ref(ret, Q, d)
    return ret

def source(Q):
    """ returns the source term S, given state Q
    """
    ret = zeros(nV)
    source_ref(ret, Q)
    return ret

def Bdot(ret, x, Q, d):
    """ ret is modified in-place, to consist of B.x,
        where B is the non-conservative matrix in direct d, given state Q
    """
    Bdot_cons(ret, x, Q, d)

def max_abs_eig(Q, d):
    """ returns the largest absolute value of the eigenvalues of the system
    """
    return max_abs_eigs(Q,d)

def system(Q, d):
    """ returns the system matrix M (where dQ/dt + M.dQ/dx = S)
    """
    raise Exception("System Jacobian not implemented in this example.\n"
                    "NOTE: cannot use Osher-Solomon flux in this example.")
