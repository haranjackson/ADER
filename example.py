import matplotlib.pyplot as plt
import numpy as np

from ader.solver import Solver


def gpr_test():

    from models.gpr.system import flux_gpr, nonconservative_matrix_gpr, source_gpr
    from models.gpr.system import system_matrix_gpr, max_eig_gpr
    from models.gpr.tests import first_stokes_problem_IC

    # Stokes' First Problem consists of the interface between flow v=-0.1 on the
    # left and v=0.1 on the right, in the y-axis.
    initial_grid, MP, tf, dX = first_stokes_problem_IC()

    # Set up the solver
    S = Solver(nvar=17, ndim=1, flux=flux_gpr,
               nonconservative_matrix=nonconservative_matrix_gpr,
               source=source_gpr, system_matrix=system_matrix_gpr,
               max_eig=max_eig_gpr, model_params=MP, ncore=4)

    # Solve for the data given
    sol = S.solve(initial_grid, tf, dX, cfl=0.9, verbose=True)

    # Plot the velocity in the y-axis
    plt.plot(np.linspace(0,1,len(initial_grid)), sol[:, 3] / sol[:, 0])
    plt.title("The GPR Model: Stokes' First Problem")
    plt.xlabel('x')
    plt.ylabel('y-velocity');

    return sol


def reactive_euler_test():

    from models.reactive_euler.system import flux_reactive_euler
    from models.reactive_euler.system import source_reactive_euler
    from models.reactive_euler.tests import shock_induced_detonation_IC

    grids = []
    def callback(u, t, count):
        grids.append(u.copy())

    initial_grid, MP, tf, dX = shock_induced_detonation_IC()

    S = Solver(nvar=6, ndim=1, flux=flux_reactive_euler,
               source=source_reactive_euler, model_params=MP)

    sol = S.solve(initial_grid, tf, dX, cfl=0.6, verbose=True, callback=callback)

    n = len(grids)
    k = 6
    for i in range(1, k+1):
        plt.plot(np.linspace(0,1,len(initial_grid)), grids[int(i*(n-1)/k)][:, 0])
    plt.title("Reactive Euler: Shock-Induced Detonation")
    plt.xlabel('x')
    plt.ylabel('density');

    return sol


if __name__ == "__main__":
    print("Uncomment the test you would like to see")
    #sol_gpr = gpr_test()
    #sol_reactive_euler, arrays_reactive_euler = reactive_euler_test()
