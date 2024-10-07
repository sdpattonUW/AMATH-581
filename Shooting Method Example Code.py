import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def bvpexam_rhs(y, x):
    return [y[1], -(x**2 - np.sin(x))*y[1] + np.cos(x)**2 * y[0] + 5]

xspan = [0, 1]
A = -3
dA = 0.5

for j in range(100):
    y0 = [3, A]
    x = np.linspace(xspan[0], xspan[1], 100)
    ysol = odeint(bvpexam_rhs, y0, x)

    if abs(ysol[-1, 1] - 5) < 10**(-6):
        break

    if ysol[-1,1] < 5:
        A += dA
    else:
        A-= dA
        dA/= 2

print(j)
plt.plot(x,ysol[:,0])
plt.show()