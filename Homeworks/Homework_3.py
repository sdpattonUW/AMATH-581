import numpy as np
from scipy.integrate import trapezoid, solve_ivp
import matplotlib.pyplot as plt

# Part A #
def shoot1(dummy, x, epsilon):
    return [x[1], (dummy**2 - epsilon)*x[0]]

tol = 1e-4
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
        #we do not use odeint anymore due to an issue with grading 
        #y = odeint(shoot1, x0, xshoot, args = (epsilon,)) 
        sol = solve_ivp(shoot1, [xshoot[0], xshoot[-1]], x0, args=(epsilon,), t_eval=xshoot, method='RK45')
        y = sol.y.T
    
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

# Part B #
from scipy.sparse.linalg import eigs
L = 4
N = 79
x = np.linspace(-L, L, N+2)
dx = x[1] - x[0]

P = np.zeros((N,N))
for j in range(N):
    P[j,j] = x[j+1] ** 2

A = np.zeros((N,N))
for j in range(N):
    A[j,j] = -2 - (x[j+1]**2 * dx**2)
for j in range(N-1):
    A[j, j + 1] = 1
    A[j + 1, j] = 1
A[0,0] = 4/3
A[0, 1] = -4/3
A[-1,-1] = 4/3
A[-1, -2] = -4/3
Amat = A / (dx**2)

linL = -Amat

D,V = eigs(linL, k = 5, sigma=0, which='LM')

V5 = np.array(V.real)
D5 = np.array(D.real)

psi0 = np.zeros(5)
psiN = psi0

for n in range(5):
    psi0[n] = ((4 * V5[0, n] - V5[1,n]) / (2*dx)) / (3/(2*dx) + np.sqrt(L**2 - D5[n]))
    psiN[n] = ((4 * V5[-1, n] - V5[-2,n]) / (2*dx)) / (3/(2*dx) + np.sqrt(L**2 - D5[n]))

V5 = np.vstack((psi0, V5, psiN))

for i in range(V5.shape[1]):
    norm_factor = np.sqrt(trapezoid(V5[:, i]**2, x))
    V5[:, i] = np.abs(V5[:, i] / norm_factor)

A3 = V5
A4 = D5

# Part C #
def shoot2(t, x, epsilon, gamma):
    return [x[1], (gamma * np.abs(x[0])**2 + t**2 - epsilon) * x[0]]

tol = 1e-4
L = 2
dx = 0.1
A_start = 1e-5
xshoot = np.arange(-L, L + dx, dx)
eigenvalues = []
eigenfunctions = []

for gamma in [0.05, -0.05]:
    epsilon_start = 0.1

    for modes in range(1, 3):
        epsilon = epsilon_start
        depsilon = 0.2

        A = A_start
        dA = 0.01

        for _ in range(100):

            for _ in range(100):
                x0 = [A, A * np.sqrt(L**2 - epsilon_start)]

                sol = solve_ivp(shoot2, [xshoot[0], xshoot[-1]], x0, t_eval=xshoot, args=(epsilon, gamma), method='RK45')

                y = sol.y.T  

                if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol:
                    break

                if ((-1) ** (modes + 1)) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            Area = trapezoid(y[:, 0] ** 2, xshoot)

            if abs(Area - 1) < tol:
                eigenvalues.append(epsilon)
                break

            if Area < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        epsilon_start = epsilon + 0.1

        norm = trapezoid(y[:, 0] ** 2, xshoot)
        eigenfunction_normalized = y[:, 0] / np.sqrt(norm)
        eigenfunctions.append(np.abs(eigenfunction_normalized))
        #print(gamma)

eigenfunctions = np.array(eigenfunctions).T
eigenvalues = np.array(eigenvalues)

A5 = eigenfunctions[:,:2]
A7 = eigenfunctions[:,2:]

A6 = eigenvalues[:2]
A8 = eigenvalues[2:]

#print(A6)
#print(A8)

#plt.figure(figsize=(10, 5))
#plt.plot(xshoot, A7[:, 0], label=r'$\phi_1$ for $\gamma = -0.05$', color="orange")
#plt.plot(xshoot, A7[:, 1], label=r'$\phi_2$ for $\gamma = -0.05$', color="green")
#plt.plot(xshoot, A5[:, 0], label=r'$\phi_1$ for $\gamma = 0.05$', color="blue", linestyle='--')
#plt.plot(xshoot, A5[:, 1], label=r'$\phi_2$ for $\gamma = 0.05$', color="red", linestyle='--')
#plt.xlabel('x')
#plt.ylabel(r'Normalized $|\phi_n|$')
#plt.legend()
#plt.title('Normalized Eigenfunctions for γ = ±0.05')
#plt.show()

# Part D #
def RHS(x, phi, E):
    return [phi[1], (x**2 - E)*phi[0]]

E = 1
K = 1
L = 2
xspan = [-L,L]
x0 = [1, np.sqrt(K*L**2 - 1)]
TOL = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
A9 = np.zeros(4)

methods = ['RK45', 'RK23', 'Radau', 'BDF']

results = {
    'RK45': {'step_sizes': [], 'local_errors': []},
    'RK23': {'step_sizes': [], 'local_errors': []},
    'Radau': {'step_sizes': [], 'local_errors': []},
    'BDF': {'step_sizes': [], 'local_errors': []}
}

for i, solver in enumerate(methods):

    for tol in TOL:
        # declare the tolerance to be the value of TOL we are iterating this time around
        options = {'rtol': tol, 'atol': tol} 
        # solve the ivp for our value of tol and our method
        sol = solve_ivp(RHS, xspan, x0, method = solver, args = (E,), **options)

        # calculate the average step size
        step_sizes = np.diff(sol.t)
        avg_step_size = np.mean(step_sizes)
        results[solver]['step_sizes'].append(avg_step_size)

        # calculate the errors
        errors = np.diff(sol.y[0])
        local_error = np.mean(np.abs(errors))
        results[solver]['local_errors'].append(local_error)
    
    slope, _ = np.polyfit(np.log(results[solver]['step_sizes']), np.log(TOL), 1)
    A9[i] = slope 
    #print(f"Slope for {solver}: {slope}")

# Plot on log-log scale for each method
#plt.figure()
#for method in methods:
#    plt.loglog(results[method]['step_sizes'], TOL, label=f'{method} Method')
#plt.xlabel('Average Step Size')
#plt.ylabel('Tolerance')
#plt.legend()
#plt.show()

# Part E #
from scipy.special import hermite

xshoot = np.arange(-4, 4.1, .1)

A10 = []
A11 = []
A12 = []
A13 = []

exact_eigenvalues = np.array([1/2, 3/2, 5/2, 7/2, 9/2])

for n in range(5):

    # Determine the exact solution using the hermite polynomials
    H_n = hermite(n)
    exact_solution = exact_solution = np.exp(-xshoot**2 / 2) * H_n(xshoot)

    exact_norm = trapezoid(exact_solution**2, xshoot)
    exact_solution = np.abs(exact_solution / np.sqrt(exact_norm))

    # Errors for A
    eigenfunction_error_a = np.sqrt(trapezoid((A1[:,n] - exact_solution)**2, xshoot))
    A10.append(eigenfunction_error_a)

    eigenvalue_error_a = 100 * (np.abs(A2[n] - exact_eigenvalues[n])/exact_eigenvalues[n])
    A11.append(eigenvalue_error_a)

    # Errors for B 
    eigenfunction_error_b = np.sqrt(trapezoid((A3[:,n] - exact_solution)**2, xshoot))
    A12.append(eigenfunction_error_a)

    eigenvalue_error_b = 100 * (np.abs(A4[n] - exact_eigenvalues[n])/exact_eigenvalues[n])
    A13.append(eigenvalue_error_a)









