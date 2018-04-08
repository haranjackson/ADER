import matplotlib.pyplot as plt
import numpy as np

from ader.solver import Solver


def gpr_test():

    from models.gpr.system import F_gpr, B_gpr, S_gpr, M_gpr, max_eig_gpr
    from models.gpr.tests import first_stokes_problem_IC
    from models.gpr.tests import first_stokes_problem_exact

    # Stokes' First Problem consists of the interface between flow v=-0.1 on
    # the left and v=0.1 on the right, in the y-axis.
    initial_grid, model_params, final_time, dX = first_stokes_problem_IC()

    # Set up the solver
    S = Solver(nvar=17, ndim=1, F=F_gpr, B=B_gpr, S=S_gpr,
               model_params=model_params, M=M_gpr, max_eig=max_eig_gpr,
               ncore=4)

    # Solve for the data given
    sol = S.solve(initial_grid, final_time, dX, cfl=0.9, verbose=True)

    # Plot the velocity in the y-axis
    n = len(initial_grid)
    x = np.linspace(0, 1, n)
    plt.plot(x, sol[:, 3] / sol[:, 0], 'x', label='solver')
    plt.plot(x, first_stokes_problem_exact(n=n), label='exact')
    plt.legend()
    plt.title("The GPR Model: Stokes' First Problem")
    plt.xlabel('x')
    plt.ylabel('y-velocity')

    return sol


def reactive_euler_test():

    from models.reactive_euler.system import F_reactive_euler
    from models.reactive_euler.system import S_reactive_euler
    from models.reactive_euler.tests import shock_induced_detonation_IC

    grids = []  # will store the grid at every timestep, for plotting later

    def callback(u, t, count):
        grids.append(u.copy())

    initial_grid, model_params, final_time, dX = shock_induced_detonation_IC()

    S = Solver(nvar=6, ndim=1, F=F_reactive_euler, S=S_reactive_euler,
               model_params=model_params)

    sol = S.solve(initial_grid, final_time, dX, cfl=0.6, verbose=True,
                  callback=callback)

    # plot density at 6 different evenly-spaced times
    x = np.linspace(0, 1, len(initial_grid))
    for i in np.linspace(0, len(grids) - 1, 6, dtype=int):
        plt.plot(x, grids[i][:, 0])
    plt.title("Reactive Euler: Shock-Induced Detonation")
    plt.xlabel('x')
    plt.ylabel('density')

    return sol


if __name__ == "__main__":
    print("Uncomment the test you would like to see")
    # sol_gpr = gpr_test()
    # sol_reactive_euler = reactive_euler_test()
