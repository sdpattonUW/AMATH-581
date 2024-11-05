import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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





