import numpy as np
from scipy.integrate import odeint, trapezoid, solve_ivp
import matplotlib.pyplot as plt

# Part A #
def shoot2(x, dummy, epsilon):
    return [x[1], (dummy**2 - epsilon)*x[0]] 

tol = 1e-4
L = 4
dx = 0.1
xshoot = np.arange(-L,L+dx,dx)
epsilon_start = 0.1
eigenvalues = []
eigenfunctions = []
x0 = [1, np.sqrt(L**2 - epsilon_start)]

for modes in range(1,6):
    epsilon = epsilon_start
    depsilon = 0.2

    for _ in range (1000):
        y = odeint(shoot2, x0, xshoot, args = (epsilon,))
    
        if abs(y[-1,1] + np.sqrt(L**2 - epsilon)*y[-1,0]) < tol:
            eigenvalues.append(epsilon)
            break

        if ((-1) ** (modes + 1))*(y[-1,1]+np.sqrt(L**2 - epsilon)*y[-1,0])  > 0:
            epsilon += depsilon

        else:
            epsilon -= depsilon
            depsilon /= 2

    epsilon_start = epsilon + 0.1

    norm = trapezoid(y[:, 0] * y[:,0], xshoot)
    eigenfunction_normalized = y[:,0] / np.sqrt(norm)
    eigenfunctions.append(np.abs(eigenfunction_normalized))

A1 = np.array(eigenfunctions).T
A2 = np.array(eigenvalues)

# Part B #
from scipy.sparse.linalg import eigs
L = 4
N = 79
x = np.linspace(-L, L, N+2)
dx = x[1] - x[0]

P = np.zeros((N,N))
for j in range(N):
    P[j,j] = x[j+1] ** 2

A = np.zeros((N,N))
for j in range(N):
    A[j,j] = -2 - (x[j+1]**2 * dx**2)
for j in range(N-1):
    A[j, j + 1] = 1
    A[j + 1, j] = 1
A[0,0] = 4/3
A[0, 1] = -4/3
A[-1,-1] = 4/3
A[-1, -2] = -4/3
Amat = A / (dx**2)

linL = -Amat

D,V = eigs(linL, k = 5, sigma=0, which='LM')

V5 = V.real
D5 = D.real

dummy = np.zeros((81,5))
dummy[1:-1, :] = V5
V5 = np.array(dummy)

for i in range(V5.shape[1]):
    norm_factor = np.sqrt(trapezoid(V5[:, i]**2, x))
    V5[:, i] = np.abs(V5[:, i] / norm_factor)

A3 = V5
A4 = D5

# Part C #
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
