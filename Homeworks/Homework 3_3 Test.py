import numpy as np
from scipy.integrate import solve_bvp, trapezoid
import matplotlib.pyplot as plt

L = 2
dx = 0.1
epsilon_guess = 0.1
gamma_values = [0.05, -0.05]
x_span = np.arange(-L, L + dx, dx)

def bvp_rhs2(x, y, p, gamma):
    epsilon = p[0]
    return np.vstack((y[1], (gamma * np.abs(y[0])**2 + (x**2) - epsilon) * y[0]))

def bvp_bc2(yl, yr, p):
    return np.array([yl[0], yl[1] - 0.1, yr[0]])

def mat4init_mode1(x):
    return np.array([np.cos((np.pi / 8) * x), -(np.pi / 2) * np.sin((np.pi / 8) * x)])

def mat4init_mode2(x):
    return np.array([np.sin((np.pi / 2) * x), (np.pi / 2) * np.cos((np.pi / 2) * x)])

def solve_mode(gamma, init_guess_func):
    init_guess = init_guess_func(x_span)
    p_initial = [epsilon_guess]

    def rhs(x, y, p):
        return bvp_rhs2(x, y, p, gamma)

    solution = solve_bvp(rhs, bvp_bc2, x_span, init_guess, p=p_initial)

    if not solution.success:
        raise RuntimeError("Solver did not converge for gamma =", gamma)
    
    # Normalize the eigenfunctions using trapezoid
    phi1 = solution.sol(x_span)[0]
    norm_phi1 = phi1 / np.sqrt(trapezoid(phi1**2, x_span))

    return np.abs(norm_phi1), solution.p[0]  # Return the normalized eigenfunction and eigenvalue

A5, A6, A7, A8 = [], [], [], []

for gamma in gamma_values:
    # Solve for the first mode
    eigenfunc1, eigenvalue1 = solve_mode(gamma, mat4init_mode1)
    # Solve for the second mode
    eigenfunc2, eigenvalue2 = solve_mode(gamma, mat4init_mode2)

    # Store results
    if gamma > 0:
        A5 = np.column_stack((eigenfunc1, eigenfunc2))  # Eigenfunctions for gamma = 0.05
        A6 = np.array([eigenvalue1, eigenvalue2])        # Eigenvalues for gamma = 0.05
    else:
        A7 = np.column_stack((eigenfunc1, eigenfunc2))  # Eigenfunctions for gamma = -0.05
        A8 = np.array([eigenvalue1, eigenvalue2])        # Eigenvalues for gamma = -0.05

# Plot the normalized eigenfunctions for visualization
plt.figure(figsize=(10, 5))
plt.plot(x_span, A7[:, 0], label=r'$\phi_1$ for $\gamma = -0.05$', color="orange")
plt.plot(x_span, A7[:, 1], label=r'$\phi_2$ for $\gamma = -0.05$', color="green")
plt.plot(x_span, A5[:, 0], label=r'$\phi_1$ for $\gamma = 0.05$', color="blue", linestyle='--')
plt.plot(x_span, A5[:, 1], label=r'$\phi_2$ for $\gamma = 0.05$', color="red", linestyle='--')
plt.xlabel('x')
plt.ylabel(r'Normalized $|\phi_n|$')
plt.legend()
plt.title('Normalized Eigenfunctions for γ = ±0.05')
plt.show()

print(A6)
print(A8)