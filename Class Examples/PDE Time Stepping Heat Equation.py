import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

tspan = np.arange(0, 2.2, 0.2)

n = 100
k = 1
L = 20

x2 = np.linspace(-L/2, L/2, n+1)
x = x2[:n]
dx = x[1] - x[0]

e1 = np.ones(n)
diagonals = [e1, -2*e1, e1]
offsets  = [-1, 0, 1]
A = diags(diagonals, offsets, shape=(n,n), format='csr')
A[0, n-1] = 1
A[n-1, 0] = 1

def rhs(u, t, k, dx, A):
    return (k / dx**2) * A.dot(u)

u0 = np.exp(-x**2)
y = odeint(rhs, u0, tspan, args=(k, dx, A))

plt.figure(figsize=(10, 6))

for i, t in enumerate(tspan):
    plt.plot(x, y[i, :], label=f't={t:.1f}')
    
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Heat Equation Solution Over Time')
plt.legend()
plt.show()

# 2D Pde Code #
Lx = 20
Ly = 20
nx = 100
ny = 100
N = nx * ny

x2 = np.linspace(-Lx/2, Lx/2, nx+1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny+1)
y = y2[:ny]

X, Y = np.meshgrid(x, y)

U  =np.exp(-X**2 -Y**2)
u = U.flatten()[:N].reshape(N,1)

u0 = U.flatten()

# Constructing 2D Laplacian operator using Kronecker products
e1 = np.ones(nx)
D1 = diags([e1, -2*e1, e1], [-1, 0, 1], shape=(nx, nx), format='csr')
D1[0, -1] = D1[-1, 0] = 1  # Periodic BC in x
A = (k / dx**2) * kron(D1, eye(ny)) + (k / dx**2) * kron(eye(nx), D1)

# Right-hand side function for ODE solver
def rhs(u, t, A):
    return A.dot(u)

# Solve the PDE over time
solution = odeint(rhs, u0, tspan, args=(A,))

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-Lx/2, Lx/2)
ax.set_ylim(-Ly/2, Ly/2)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')
ax.set_title('2D Heat Equation Solution Over Time')

# Function to update the surface plot
def update_plot(frame):
    ax.clear()
    Z = solution[frame].reshape((nx, ny))
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_zlim(0, 1)
    ax.set_title(f"Time = {tspan[frame]:.1f}")
    return surf,

# Animation
anim = FuncAnimation(fig, update_plot, frames=len(tspan), blit=False)

# Display animation
plt.show()
