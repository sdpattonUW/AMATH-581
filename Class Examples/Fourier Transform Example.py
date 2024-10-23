import numpy as np
import matplotlib.pyplot as plt

L = 20 # define the computational domain [-L/2, L/2]
n = 128 # define the number of Fourier modes 2^n
x2 = np.linspace(-L/2, L/2, n+1) # Define the domain
x = x2[:n] # Consider only the first n points

u = np.exp(-x * x) # gaussian
ut = np.fft.fft(u) # fft of gaussian -> another gaussian
utshift = np.fft.fftshift(ut)

plt.figure(1)
plt.plot(x ,u)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Initial Gaussian')
plt.show()

plt.figure(2)
plt.plot(x ,np.abs(ut))
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Initial Gaussian')
plt.show()

plt.figure(3)
plt.plot(x ,np.abs(utshift))
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Initial Gaussian')
plt.show()