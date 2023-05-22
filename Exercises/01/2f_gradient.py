import numpy as np
import matplotlib.pyplot as plt

# usamos numpy linspace para generar varios numeros dentro del rango
# pero espaciados equidistantemente (para que las curvas sean mas suaves)
x = np.linspace(-5, 5, 10)  # 10 numeros es la consigna inicial
# print(x)  [-5. -3.88888889 -2.77777778 -1.66666667 -0.55555556  0.55555556  1.66666667  2.77777778  3.88888889  5]

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

# convertimos las listas en arrays numpy
ny1 = np.array(y1)
ny2 = np.array(y2)
ny3 = np.array(y3)
ny4 = np.array(y4)
ny5 = np.array(y5)
ny6 = np.array(y6)

# obtenemos las aproximaciones a las pendientes usando gradient (calcula derivadas de cada punto = pendientes)
g1 = np.gradient(ny1) # [-26.2962963  -22.59259259 -15.18518519  -7.77777778  -0.37037037  7.03703704  14.44444444  21.85185185  29.25925926  32.96296296]
g2 = np.gradient(ny2) # lineal [3.33333333 3.33333333 3.33333333 3.33333333 3.33333333 3.33333333 3.33333333 3.33333333 3.33333333 3.33333333]
g3 = np.gradient(ny3)
g4 = np.gradient(ny4)
g5 = np.gradient(ny5)
g6 = np.gradient(ny6)

# print(g1, g2, g3, g4, g5, g6)

# ahora iteramos por las pendientes e identificamos cuando cambia de signo
derivada_anterior = 0
signo_anterior = 0
counter = 0

# esto es para la funcion 1
# y1 valores de y para la funcion 1
# ny1 valores de y en forma de array numpy
# g1 valores de pendientes de la funcion 1 (gradient)0

# Iteramos sobre las pendientes
for derivada in g1:
    counter += 1
    if derivada < 0:
        signo = "-"
    else:
        signo = "+"

    # Si todavía no guardé el signo (sigue con inicializador 0), guardo signo y derivada actual para comparar con futuras
    if signo_anterior == 0:
        signo_anterior = signo
        derivada_anterior = derivada
        continue
    else:
        # Si el signo actual es igual al signo anterior, no hago nada
        if signo_anterior == signo:
            continue
        # Si el signo actual es diferente al signo anterior, se produce un cambio de signo
        else:
            print("Anterior", "\nPendiente:", g1[counter - 2], "\nX:", x[counter - 2], "\n\nPosterior", "\nPendiente:",
            derivada, "\nX:", x[counter - 1])
            signo_anterior = signo


# haciendo lo mismo para las funciones 2, 3, 4, 5, y 6 (g2, g3, g4, g5, g6)
# se obtienen las respuestas para las otras funciones

# usando este mismo codigo con distintos parametros de linspace (10, 50, 100, 1000)
# se puede responder la pregunta sobre aproximación a cero de las pendientes

# en este codigo si queremos ver las pendientes de la funcion f1
# usamos las variables y1 (valores de y para f1)
# y g1 (valores de pendientes o derivadas para f1)
# si quieren ver como dan f2, f3, f4 ...
# reemplazar las variables (y2, g2; y3, g3, etc.)

# inicializamos figura
fig = plt.figure()

# este es el numero de puntos que tenemos (10, 50, 100, ...)
# para este ejemplo empezamos por 10
data_points = 10

for i in range(0, data_points):
    plot = i + 1
    # generamos un sistema de ejes para cada data point
    ax = fig.add_subplot(2, 5, plot, xbound=[-5, 5])
    # ploteamos la funcion original, en este caso f1
    ax.plot(x, y1)
    ax.set_ylim(np.amin(y1), np.amax(y1))
    ax.grid()

    # y además ploteamos las distintas pendientes
    # para eso usamos la pendiente i que nos devuelve np.gradient
    # y al menos dos puntos en x
    # cuales dos puntos? bueno puede ser buena idea tomar el punto x-1 y x+1 con respecto al punto x
    # pero ojo! en el primer data point no existe x-1, y en el ultimo no existe x+1 !
    if i != 0 and i < (data_points - 1):
        # generamos un miniarray de valores en x de 3 puntos para graficar las rectas
        rx = np.linspace(x[i - 1], x[i + 1], 3)
        a = g1[i]  # el valor i del array np.gradient (la pendiente)
        b = 1  # un valor de ordenada al origen cualquiera
        ry = a * rx + b  # nuestra funcion de la recta i
        ax.plot(rx, ry, color='tab:red', linewidth=2)

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
plt.show()