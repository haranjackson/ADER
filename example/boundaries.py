from options import ndim
from solvers.misc import extend


def transmissive_BC(u, reflect=0):
    ret = extend(u, 1, 0)
    if ndim > 1:
        ret = extend(ret, 1, 1)
        if ndim > 2:
            ret = extend(ret, 1, 2)
    return ret


def periodic_BC(u):
    ret = extend(u, 1, 0)
    ret[0] = ret[-2]
    ret[-1] = ret[1]
    if ndim > 1:
        ret = extend(ret, 1, 1)
        ret[:, 0] = ret[:, -2]
        ret[:, -1] = ret[:, 1]
    if ndim > 2:
        ret = extend(ret, 1, 2)
        ret[:, :, 0] = ret[:, :, -2]
        ret[:, :, -1] = ret[:, :, 1]
    return ret
