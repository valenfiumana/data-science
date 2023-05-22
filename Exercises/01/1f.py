import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()

x = np.linspace(-5, 5, 25)

# FUNCION 1: y=3x^2+3x−6
def funcion1(x):
    y = 3 * x ** 2 + 3 * x - 6
    return y

# FUNCIÓN 2: y=3x−6
def funcion2(x):
    y = 3 * x - 6
    return y

# FUNCIÓN 3: y=x^3
def funcion3(x):
    y = x ** 3
    return y

# FUNCIÓN 4: y=1 / (1+e^−x)
def funcion4(x):
    y = 1 / (1 + np.exp(-x))
    return y

# FUNCIÓN 5: y=1/2 * x + 1
def funcion5(x):
    y = 1/2 * x + 1
    return y


# FUNCIÓN 6: y=x^2+1
def funcion6(x):
    y = x**2+ 1
    return y

# GRAPHS
plt.plot(x, funcion1(x), color='maroon', label='y=3x^2+3x-6')
plt.plot(x, funcion2(x), color='green', label='y=3x-6')
plt.plot(x, funcion3(x), color='aquamarine', label='y=x^3')
plt.plot(x, funcion4(x), color='deeppink', label='y=1/(1+e^-x)')
plt.plot(x, funcion5(x), color='gold', label='y=1/2*x+1')
plt.plot(x, funcion6(x), color='yellowgreen', label='y=x^2+1')

plt.grid()
plt.legend()
plt.show()

