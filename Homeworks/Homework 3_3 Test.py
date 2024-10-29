import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt

# Part A #
def shoot2(t, x, epsilon, gamma):
    return [x[1], (gamma * np.abs(x[0])**2 + t**2 - epsilon) * x[0]]

tol = 1e-4
L = 2
dx = 0.1
A_start = 1e-5
xshoot = np.arange(-L, L + dx, dx)
eigenvalues = []
eigenfunctions = []

for gamma in [0.05, -0.05]:
    epsilon_start = 0.1

    for modes in range(1, 3):
        epsilon = epsilon_start
        depsilon = 0.2

        A = A_start
        dA = 0.01

        for _ in range(100):

            for _ in range(100):
                x0 = [A, A * np.sqrt(L**2 - epsilon_start)]

                # Using solve_ivp with "RK45" method (similar to odeint's default)
                sol = solve_ivp(shoot2, [xshoot[0], xshoot[-1]], x0, t_eval=xshoot, args=(epsilon, gamma), method='RK45')

                y = sol.y.T  # transpose to make it compatible with odeint's output

                if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol:
                    break

                if ((-1) ** (modes + 1)) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            Area = trapezoid(y[:, 0] ** 2, xshoot)

            if abs(Area - 1) < tol:
                eigenvalues.append(epsilon)
                break

            if Area < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        epsilon_start = epsilon + 0.1

        norm = trapezoid(y[:, 0] ** 2, xshoot)
        eigenfunction_normalized = y[:, 0] / np.sqrt(norm)
        eigenfunctions.append(np.abs(eigenfunction_normalized))
        print(gamma)

A1 = np.array(eigenfunctions).T
A2 = np.array(eigenvalues)

print(A2)

plt.plot(xshoot, A1[:, 0], '--')
plt.plot(xshoot, A1[:, 1], '--')
plt.plot(xshoot, A1[:, 2], color='red')
plt.plot(xshoot, A1[:, 3], color='blue')
plt.legend(['1st + mode','2nd + mode','1st - mode','2nd - mode'])
plt.show()
