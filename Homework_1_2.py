import numpy as np

A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2, 0, -3],[0, 0, -1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1, 2, -1])

A4 = A + B
A5 = 3 * x - 4 * y
A6 = A@x
A7 = B@(x-y)
A8 = D@x
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