from itertools import product

from solvers.fv.fv import fv_launcher
from solvers.dg.dg import dg_launcher
from solvers.weno.weno import weno_launcher
from system import max_abs_eigs
from options import nx, ny, nz, CFL, dx, dy, dz, ndim


def ader_stepper(pool, u, BC, dt):

    wh = weno_launcher(BC(u))
    qh = dg_launcher(pool, wh, dt)
    u += fv_launcher(pool, qh, dt)


def timestep(u, count, t, tf):
    """ Calculates dt, based on the maximum wavespeed across the domain
    """
    MAX = 0
    for i, j, k in product(range(nx), range(ny), range(nz)):

        Q = u[i, j, k]
        MAX = max(MAX, max_abs_eigs(Q, 0) / dx)
        if ndim > 1:
            MAX = max(MAX, max_abs_eigs(Q, 1) / dy)
            if ndim > 2:
                MAX = max(MAX, max_abs_eigs(Q, 2) / dz)

    dt = CFL / MAX
    if count <= 5:
        dt *= 0.2
    if t + dt > tf:
        return tf - t
    else:
        return dt
