from example.gpr.systems.conserved import flux_cons, block_cons, source_cons, system_cons
from example.gpr.systems.eigenvalues import max_abs_eigs


def flux(ret, Q, d):
    """ ret is modified in-place, to consist of the flux F, in direction d,
        given state Q
    """
    # the flux for the GPR model - replace with your own
    flux_cons(ret, Q, d)


def block(ret, Q, d):
    """ ret is modified in-place, to consist of the non-conservative matrix B,
        in direction d, given state Q
    """
    # the block matrix for the GPR model - replace with your own
    block_cons(ret, Q, d)


def source(ret, Q):
    """ ret is modified in-place, to consist of the source term S,
        given state Q
    """
    # the source for the GPR model - replace with your own
    source_cons(ret, Q)


def max_abs_eig(Q, d):
    """ returns the largest absolute value of the eigenvalues of the system
    """
    # the maximum eigenvalue for the GPR model - replace with your own
    return max_abs_eigs(Q, d)


def system(Q, d):
    """ returns the system matrix M (where dQ/dt + M.dQ/dx = S)
    """
    # the system jacobian for the GPR model - replace with your own
    return system_cons(Q, d)
