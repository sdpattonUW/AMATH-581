import numpy as np
import matplotlib.pyplot as plt

# Part I #
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

# Part II #

A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2, 0, -3],[0, 0, -1]])
D = np.array([[1,2],[2,3],[-1,0]])
x1 = np.array([1,0])
y = np.array([0,1])
z = np.array([1, 2, -1])

A4 = A + B
A5 = 3 * x1 - 4 * y
A6 = A@x1
A7 = B@(x1-y)
A8 = D@x1
A9 = D@y + z
A10 = A@B
A11 = B@C
A12 = C@D

print("A4 =\n", A4)
print("A5 =\n", A5)
print("A6 =\n", A6)
print("A7 =\n", A7)
print("A8 =\n", A8)
print("A9 =\n", A9)
print("A10 =\n", A10)
print("A11 =\n", A11)
print("A12 =\n", A12)