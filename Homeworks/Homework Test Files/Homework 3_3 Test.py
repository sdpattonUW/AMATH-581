import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, solve_ivp

def shoot2(t, x, epsilon, gamma):
    return [x[1], (gamma * np.abs(x[0])**2 + t**2 - epsilon) * x[0]]

tol = 1e-4
L = 2
dx = 0.1
A_start = 0.01
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
                x0 = [A, A * np.sqrt(L**2 - epsilon)]

                sol = solve_ivp(shoot2, [xshoot[0], xshoot[-1]], x0, t_eval=xshoot, args=(epsilon, gamma), method='RK45')

                y = sol.y.T  

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
        #print(gamma)

eigenfunctions = np.array(eigenfunctions).T
eigenvalues = np.array(eigenvalues)

A5 = eigenfunctions[:,:2]
A7 = eigenfunctions[:,2:]

A6 = eigenvalues[:2]
A8 = eigenvalues[2:]

print(A6)
print(A8)

plt.figure(figsize=(10, 5))
plt.plot(xshoot, A7[:, 0], label=r'$\phi_1$ for $\gamma = -0.05$', color="orange")
plt.plot(xshoot, A7[:, 1], label=r'$\phi_2$ for $\gamma = -0.05$', color="green")
plt.plot(xshoot, A5[:, 0], label=r'$\phi_1$ for $\gamma = 0.05$', color="blue", linestyle='--')
plt.plot(xshoot, A5[:, 1], label=r'$\phi_2$ for $\gamma = 0.05$', color="red", linestyle='--')
plt.xlabel('x')
plt.ylabel(r'Normalized $|\phi_n|$')
plt.legend()
plt.title('Normalized Eigenfunctions for γ = ±0.05')
plt.show()