from numpy import inf
from numpy.linalg import det

from models.gpr.misc.functions import sigma_norm
from models.gpr.variables.state import sigma


def theta1inv(ρ, A, MP):
    """ Returns 1/θ1
    """
    τ1 = MP.τ1

    if τ1 == inf:
        return 0

    cs2 = MP.cs2

    if MP.PLASTIC:
        σY = MP.σY
        n = MP.n
        σ = sigma(ρ, A, MP)
        sn = sigma_norm(σ)
        sn = min(sn, 1e8)   # Hacky fix
        return 3 * det(A)**(5 / 3) / (cs2 * τ1) * (sn / σY) ** n
    else:
        return 3 * det(A)**(5 / 3) / (cs2 * τ1)


def theta2inv(ρ, T, MP):
    """ Returns 1/θ2
    """
    ρ0 = MP.ρ0
    T0 = MP.T0
    cα2 = MP.cα2
    τ2 = MP.τ2

    return 1 / (cα2 * τ2 * (ρ / ρ0) * (T0 / T))
