import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot2(x, dummy, n0, beta):
    return [x[1], (beta-n0)*x[0]]

xshoot = np.linspace(-1, 1, 1000)
n0 = 100
beta_start = n0
x0 = [0, 1]
tol = 1e-4

for modes in range(1,2):
    beta = beta_start
    dbeta = 1
    for j in range (1000):
        y = odeint(shoot2, x0, xshoot, args=(n0, beta))

        if abs(y[-1,0] - 0) < tol:
            break

        if (-1)**(modes + 1) * y[-1,0] > 0:
            beta -= dbeta
        else: 
            beta += dbeta / 2
            dbeta /= 2
        
norm = np.trapz(y[:,0] * y[:,0], xshoot )
plt.plot(xshoot, y[:,0] / np.sqrt(norm))
plt.show()
