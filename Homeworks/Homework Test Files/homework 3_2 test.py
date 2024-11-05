import numpy as np
from scipy.integrate import trapezoid, solve_ivp
import matplotlib.pyplot as plt

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

V5 = np.array(V.real)
D5 = np.array(D.real)

psi0 = np.zeros(5)
psiN = psi0

for n in range(5):
    psi0[n] = ((4 * V5[0, n] - V5[1,n]) / (2*dx)) / (3/(2*dx) + np.sqrt(L**2 - D5[n]))
    psiN[n] = ((4 * V5[-1, n] - V5[-2,n]) / (2*dx)) / (3/(2*dx) + np.sqrt(L**2 - D5[n]))

V5 = np.vstack((psi0, V5, psiN))

for i in range(V5.shape[1]):
    norm_factor = np.sqrt(trapezoid(V5[:, i]**2, x))
    V5[:, i] = np.abs(V5[:, i] / norm_factor)

plt.plot(x,V5)
plt.show()

A3 = V5
A4 = D5

