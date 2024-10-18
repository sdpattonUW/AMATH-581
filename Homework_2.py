import numpy as np
from scipy.integrate import odeint, trapezoid
import matplotlib.pyplot as plt

def shoot2(x, dummy, epsilon):
    return [x[1], (epsilon - dummy**2)*x[0]]

tol = 1e-6
L = 4.0
dx = 0.1
n0 = 100
xshoot = np.arange(-L, L+dx, dx)
epsilon_start = 1
eigenvalues = []
eigenfunctions = []
x0 = [1, np.sqrt((L)**2 - epsilon_start)]

for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = n0/100

    x0 = [1, np.sqrt((L)**2 - epsilon_start)]
    for _ in range (1000):
        y = odeint(shoot2, x0, xshoot, args=(epsilon,))

        if abs(y[-1,1] - (np.sqrt(L**2 - epsilon) * y[-1,0])) < tol:
            eigenvalues.append(epsilon)
            break

        if (-1) ** (modes + 1) * (y[-1,1] + np.sqrt(L**2 - epsilon) * y[-1,0]) > 0:
            epsilon -= depsilon

        else:
            epsilon += depsilon
            depsilon /= 2

    epsilon_start = epsilon - 0.1

    norm = trapezoid(y[:, 0] ** 2, xshoot)
    eigenfunction_normalized = np.abs(y[:,0]) / np.sqrt(norm)
    eigenfunctions.append(eigenfunction_normalized)
    

A1 = np.array(eigenfunctions).T
A1 = np.abs(A1)
A2 = np.array(eigenvalues)
