import numpy as np
import matplotlib.pyplot as plt

# Problem 1
# Newton Raphson
x = np.array([-1.6]) # Initial Guess

for j in range(1000):
    x = np.append(
        x, x[j] - (x[j] * np.sin(3 * x[j]) - np.exp(x[j])) 
        / (np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j])))
    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j])

    if abs(fc) < 1e-6:
        break

A1 = x

# Bisection Method
xl = -0.7; xr = -0.4
A2 = np.array([])
for j in range(0,100):
    xc = (xr+xl)/2
    fc = xc * np.sin(3 * xc) - np.exp(xc)
    if ( fc > 0 ):
        xl = xc
    else:
        xr = xc
    A2 = np.append(A2, xc)

    if ( abs(fc) < 1e-6 ):
        break

A3 = np.array([A1.size - 1, A2.size])

print("A1 =", A1)
print("A2 =", A2)
print("A3 =", A3)
