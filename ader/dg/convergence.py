from numpy import absolute


def blowup(qNew, max_size):
    """ Check whether qNew has blown up larger than max_size
    """
    return (absolute(qNew) > max_size).any()


def unconverged(q, qNew, TOL):
    """ Mixed convergence condition
    """
    return (absolute(q - qNew) > TOL * (1 + absolute(q))).any()
