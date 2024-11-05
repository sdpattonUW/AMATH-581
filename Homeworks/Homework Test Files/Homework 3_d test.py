import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation
def shoot2(t, x, epsilon):
    return [x[1], (t**2 - epsilon) * x[0]]

# Parameters
epsilon = 1
K = 1
L = 2
xspan = [-L, L]
x0 = np.array([1, np.sqrt(K * L**2 - 1)])
TOL_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
A9 = np.zeros((4,1))

# Initialize lists to store step sizes and slopes for each method
methods = ['RK45', 'RK23', 'Radau', 'BDF']
step_sizes = {method: [] for method in methods}

# Perform the convergence study for each method
for TOL in TOL_values:
    options = {'rtol': TOL, 'atol': TOL}
    for method in methods:
        sol = solve_ivp(shoot2, xspan, x0, method=method, args=(epsilon,), **options)
        avg_step_size = np.mean(np.diff(sol.t))  # Calculate the average step size
        step_sizes[method].append(avg_step_size)

# Plotting and computing slopes
plt.figure(figsize=(10, 8))

for i, method in enumerate(methods):
    # Log-log plot
    plt.loglog(step_sizes[method], TOL_values, label=method, marker='o')
    
    # Compute slope using polyfit
    log_step_sizes = np.log(step_sizes[method])
    log_tolerances = np.log(TOL_values)
    slope, _ = np.polyfit(log_step_sizes, log_tolerances, 1)

    A9[i,0] = slope
    
    print(f"Slope for {method}: {slope}")

print(A9)

# Set plot labels
plt.xlabel("Average Step Size (log scale)")
plt.ylabel("Tolerance (log scale)")
plt.title("Convergence Study: Step Size vs. Tolerance")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
