import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

m = 8
n = m**2
d = 20/8

e0 = np.zeros((n,1))
e1 = np.ones((n,1))
e2 = np.copy(e1)
e4 = np.copy(e0)

for j in range(1, m+1):
    e2[m*j-1] = 0  
    e4[m*j-1] = 1  

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]


diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

A = spdiags(diagonals, offsets, n, n).toarray()

A /= d**2

diagonals_B = [e1.flatten(), -1 * e1.flatten(), e1.flatten(), -1 * e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]

B = spdiags(diagonals_B, offsets_B, n, n).toarray()
B /= 2*d

diagonals_C = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_C = [-7, -1, 1, 7]
C = spdiags(diagonals_C, offsets_C,m,m).toarray()
C /= 2*d

I = np.eye(8)
C = np.kron(I,C)

plt.imshow(C)
plt.colorbar()
plt.show()

A1 = A
A2 = B
A3 = C
