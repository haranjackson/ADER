import matplotlib.pyplot as plt

from joblib import Parallel

from solvers.iterator import ader_stepper, timestep
from options import NCORE


def main(u, tf, BC, pool):

    t = 0
    count = 0

    while t < tf:

        dt = timestep(u, count, t, tf)

        ader_stepper(pool, u, BC, dt)

        t += dt
        count += 1

        print(count, ': t =', t)

    return u


if __name__ == "__main__":

    # Test taken from DOI:10.1016/j.jcp.2016.02.015
    from example.test import viscous_shock_IC
    from example.boundaries import transmissive_BC

    u, tf = viscous_shock_IC()
    pool = Parallel(n_jobs=NCORE)
    u = main(u, tf, transmissive_BC, pool)

    plt.figure(1)
    plt.plot(u[:,0,0,0])
    plt.title('density')
    plt.figure(2)
    plt.plot(u[:,0,0,2]/u[:,0,0,0])
    plt.title('velocity')
