import numpy as np
from scipy.integrate import trapezoid, solve_ivp
from scipy.special import hermite
import matplotlib.pyplot as plt

# Part A - Numerical solution parameters
def shoot2(dummy, x, epsilon):
    return [x[1], (dummy**2 - epsilon) * x[0]]

tol = 1e-4
L = 4
dx = 0.1
xshoot = np.arange(-L, L + dx, dx)
epsilon_start = 0.1
eigenvalues = []
eigenfunctions = []
x0 = [1, np.sqrt(L**2 - epsilon_start)]

for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = 0.2

    for _ in range(1000):
        sol = solve_ivp(shoot2, [xshoot[0], xshoot[-1]], x0, args=(epsilon,), t_eval=xshoot, method='RK45')
        y = sol.y.T

        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol:
            eigenvalues.append(epsilon)
            break

        if ((-1) ** (modes + 1)) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2

    epsilon_start = epsilon + 0.1

    norm = trapezoid(y[:, 0] * y[:, 0], xshoot)
    eigenfunction_normalized = y[:, 0] / np.sqrt(norm)
    eigenfunctions.append(np.abs(eigenfunction_normalized))

# Convert lists to arrays for plotting
A1 = np.array(eigenfunctions).T
A2 = np.array(eigenvalues)

# Part B - Plotting and comparing with exact Hermite polynomial solutions
