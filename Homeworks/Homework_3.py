import numpy as np
from scipy.integrate import odeint, trapezoid
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
def shoot3(x, dummy, epsilon, gamma):
    return [x[1], (gamma * abs(x[0])**2 + dummy**2 - epsilon) * x[0]] 

tol = 1e-4
L = 2
dx = 0.1
xshoot = np.arange(-L,L+dx,dx)
epsilon_start = 0.1

eigenvalues_pos = []
eigenfunctions_pos = []
eigenvalues_neg = []
eigenfunctions_neg = []

x0 = [1, np.sqrt(L**2 - epsilon_start)]

for gamma in [0.05, -0.05]:
    for modes in range(1,3):
        epsilon = epsilon_start
        depsilon = 0.1

        for _ in range(1000):
            y = odeint(shoot3, x0, xshoot, args=(epsilon, gamma))

            if abs(y[-1,0]) < tol:
                if gamma == 0.05:
                    eigenvalues_pos.append(epsilon)
                    norm = trapezoid(y[:, 0] * y[:, 0], xshoot)
                    eigenfunction_normalized = y[:, 0] / np.sqrt(norm)
                    eigenfunctions_pos.append(np.abs(eigenfunction_normalized))
                else:
                    eigenvalues_neg.append(epsilon)
                    norm = trapezoid(y[:, 0] * y[:, 0], xshoot)
                    eigenfunction_normalized = y[:, 0] / np.sqrt(norm)
                    eigenfunctions_neg.append(np.abs(eigenfunction_normalized))
                break

            if ((-1) ** (modes + 1)) * y[-1,0] > 0:
                epsilon += depsilon
            else:
                epsilon -= depsilon
                depsilon /= 2

        epsilon_start = epsilon + depsilon


plt.plot(xshoot, np.array(eigenfunctions_pos).T, xshoot, np.array(eigenfunctions_neg).T)
plt.show()
