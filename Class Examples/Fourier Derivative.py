import numpy as np
import matplotlib.pyplot as plt

def sech(x):
    return 1 / np.cosh(x)

def tanh(x):
    return np.sinh(x) / np.cosh(x)

L  = 20
n = 128
x2 = np.linspace(-L/2, L/2, n+1)
x = x2[:n]

u = sech(x)
ut = np.fft.fft(u)

k = (2 * np.pi / L) * np.concatenate((np.arange(0, n//2), np.arange(-n//2, 0)))
ut1 = 1j * k * ut # first derivative 
ut2 = -k**2 * ut # second derivative
ut3 = -1j * k**3 * ut  # third derivative

u1 = np.fft.ifft(ut1) # Inverse transform
u2 = np.fft.ifft(ut2)
u3 = np.fft.ifft(ut3)

# Analytic first derivative
u1exact = -sech(x) * tanh(x)

# Plot
plt.plot(x, u, 'r', label='Original')
plt.plot(x, u1, 'g', label='First Derivative (Approx.)')
plt.plot(x, u1exact, 'go', label='First Derivative (Exact)')
plt.plot(x, u2, 'b', label='Second Derivative')
plt.plot(x, u3, 'c', label='Third Derivative')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Function and its Derivatives')
plt.legend()
plt.grid(True)
plt.show()