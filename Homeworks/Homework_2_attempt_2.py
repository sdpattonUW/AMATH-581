import numpy as np
from scipy.integrate import odeint, trapezoid
import matplotlib.pyplot as plt



def shoot2(x, dummy, epsilon):

    return [x[1], (dummy**2 - epsilon)*x[0]] 



tol = 1e-6
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

print(A2)

plt.figure(figsize=(10, 6))
for i in range(A1.shape[1]):
    plt.plot(xshoot, A1[:, i], label=f'Eigenfunction {i+1}')
plt.xlabel('x')
plt.ylabel('Eigenfunction Amplitude')
plt.title('Normalized Eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()






