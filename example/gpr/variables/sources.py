from numpy.linalg import det

from example.parameters import ρ0, T0, cs, α, τ1, τ2


def theta1inv(ρ, A):
    """ Returns 1/θ1
    """
    return 3 * det(A)**(5/3) / (cs**2 * τ1)

def theta2inv(ρ, T):
    """ Returns 1/θ2
    """
    return 1 / (α**2 * τ2 * (ρ / ρ0) * (T0 / T))
