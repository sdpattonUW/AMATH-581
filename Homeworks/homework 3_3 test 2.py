import numpy as np
from scipy.integrate import odeint, trapezoid
import matplotlib.pyplot as plt

# Part A #
def shoot2(x, dummy, epsilon, gamma):
    return [x[1], (gamma * np.abs(x[0])**2 + dummy**2 - epsilon)*x[0]] 

tol = 1e-4
L = 2
dx = 0.1
xshoot = np.arange(-L,L+dx,dx)
eigenvalues = []
eigenfunctions = []
x0 = [0, 1e-5]

for gamma in [0.05, -0.05]:
    epsilon_start = 0.1

    for modes in range(1,3):
        epsilon = epsilon_start
        depsilon = 0.2

        for _ in range (1000):
            y = odeint(shoot2, x0, xshoot, args = (epsilon, gamma,))
        
            if abs(y[-1,0]) < tol:
                eigenvalues.append(epsilon)
                break

            if ((-1) ** (modes + 1))*(y[-1,0])  > 0:
                epsilon += depsilon

            else:
                epsilon -= depsilon
                depsilon /= 2

        epsilon_start = epsilon + 0.2

        norm = trapezoid(y[:, 0] * y[:,0], xshoot)
        eigenfunction_normalized = y[:,0] / np.sqrt(norm)
        eigenfunctions.append(np.abs(eigenfunction_normalized))
        print(gamma)

A1 = np.array(eigenfunctions).T
A2 = np.array(eigenvalues)

print(A2)

plt.plot(xshoot,A1[:,0])
plt.plot(xshoot,A1[:,1])
plt.plot(xshoot,A1[:,2],'--', color = 'red')
plt.plot(xshoot,A1[:,3],'--', color ='blue')
plt.show()