import math
import numpy as np

def equation_1(x):
  return 3 * x**2 + 3 * x - 6

def equation_2(x):
  return 3 * x - 6

def equation_3(x):
  return x ** 3

def equation_4(x):
  # usamos el numero e (Euler) de
  return 1 / (1 + np.exp(-x))

def equation_5(x):
  return 1/2*x+1

def equation_6(x):
  return x**2+1

# usamos numpy linspace para generar varios numeros dentro del rango
# pero espaciados equidistantemente (para que las curvas sean mas suaves)
x = np.linspace(-5, 5, 50) # 50 numeros deberian ser suficiente

y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []

for val in x:
  y1.append(equation_1(val))
  y2.append(equation_2(val))
  y3.append(equation_3(val))
  y4.append(equation_4(val))
  y5.append(equation_5(val))
  y6.append(equation_6(val))

import matplotlib.pyplot as plt
fig = plt.figure()

ax1 = fig.add_subplot(2,3,1, xbound=[-5, 5])
ax1.grid()

ax2 = fig.add_subplot(232, sharex=ax1) # share the x-axis with the subplot ax1
ax2.grid()

ax3 = fig.add_subplot(233, sharex=ax1)
ax3.grid()

ax4 = fig.add_subplot(234, sharex=ax1)
ax4.grid()

ax5 = fig.add_subplot(235, sharex=ax1)
ax5.grid()

ax6 = fig.add_subplot(236, sharex=ax1)
ax6.grid()

ax1.plot(x,y1, color='maroon', label='y=3x^2+3x-6')
ax2.plot(x,y2, color='green', label='y=3x-6')
ax3.plot(x,y3, color='aquamarine', label='y=x^3')
ax4.plot(x,y4, color='deeppink', label='y=1/(1+e^-x)')
ax5.plot(x,y5, color='gold', label='y=1/2*x+1')
ax6.plot(x,y6, color='yellowgreen', label='y=x^2+1')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()

plt.show()