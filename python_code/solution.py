import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the parameters
Cp = 1050  # J/mol-K
dhr = 103  # kJ/mol-K
R = 8.3144  # J/mol-K
E = 223.15  # kJ/mol
ko = 9.47e11  # min^-1
nA0 = 1  # mol
U = 150  # J/min-K
T0 = 298  # K
Tf = 368  # K
Cv = Cp - R

# Define the system of differential equations


def model(y, t):
    T, XA = y
    Q = U * (Tf - T)
    dTdt = (Q - (dhr - R * T) * ko * np.exp(-E / T) * nA0 * (1 - XA)) / Cv
    dXAdt = ko * np.exp(-E / T) * (1 - XA)
    return [dTdt, dXAdt]


# Define the initial condition
y0 = [T0, 0]

# Define the time span for integration
tspan = np.linspace(0, 180, 1000)

# Solve the system of differential equations
y = odeint(model, y0, tspan)

# Extract the solution for T and XA
T = y[:, 0]
XA = y[:, 1]

# Plot XA against T
plt.figure()
plt.plot(T, XA)
plt.xlabel('T (K)')
plt.ylabel('XA')
plt.title('XA vs T')

# Plot XA against t
plt.figure()
plt.plot(tspan, XA)
plt.xlabel('t (min)')
plt.ylabel('XA')
plt.title('XA vs t')

# Plot t against T
plt.figure()
plt.plot(tspan, T)
plt.xlabel('t (min)')
plt.ylabel('T (K)')
plt.title('T vs t')

plt.show()
