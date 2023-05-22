# Los resultados de un examen, cuya nota máxima posible es 100, siguen una distribución normal con media 68 y desviación típica 12.
# * Realice un gráfico de esta distribución normal para el intervalo
# * ¿Cuál es la probabilidad de que una persona que se presenta el examen obtenga una calificación superior a 72?
# * Calcular la proporción de estudiantes que tienen puntuaciones que exceden por lo menos en cinco puntos de la puntuación que marca la frontera entre el Apto y el No-Apto; sabiendo que son declarados No-Aptos el 25% de los estudiantes que obtuvieron las puntuaciones más bajas.
# * Si se sabe que la calificación de un estudiante es mayor que 72, ¿cuál es la probabilidad de que su calificación sea, de hecho, superior a 84?

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# GRÁFICO DISTRIBUCIÓN NORMAL
mu=68 # Mean of the Gaussian distribution
sigma=12.0 # Standard deviation of the Gaussian distribution
mi_gaussiana=norm(loc=mu,scale=sigma)

x = np.linspace(8.0,120,100) # Range of values for x-axis

plt.figure(figsize = (10, 5))
plt.plot(x,mi_gaussiana.pdf(x))
plt.title(f"Distribución Gaussiana $N(\mu,\sigma^2)$")
plt.xlabel('x')
plt.ylabel('Probabilidad')
plt.show()

# PROBA DE CALIFICACION > 72
from scipy.integrate import quad

x_lower = 72  # Lower limit
x_upper = np.inf  # Upper limit

val, abserr = quad(mi_gaussiana.pdf, x_lower, x_upper)
print('Proba de calificacion mayor a 72: ' + str(round(val, 4)))

# Tambien podemos usar sf de Scipy
# print(mi_gaussiana.sf(72))


# PROPORCION DE ESTUDIANTES
# "Calcular la proporción de estudiantes que tienen puntuaciones que exceden por lo menos en cinco puntos de la puntuación que marca la frontera entre el Apto y el No-Apto;
# sabiendo que son declarados No-Aptos el 25% de los estudiantes que obtuvieron las puntuaciones más bajas".

# Primero calculemos la calificación que engloba al 25% de notas más bajas, o sea, que corresponde al primer cuartil:
value = mi_gaussiana.ppf(0.25)
print('El 25% de las calificaciones se encuentran por debajo de ' + str(round(value, 2)))

# Ahora queremos saber la proporción de estudiantes que tienen puntuaciones que exceden en 5 puntos esa frontera
print('Hay una probabilidad de ' + str(round(mi_gaussiana.sf(value + 5), 4)) + ' de obtener calificacion '+ str(round(value + 5, 2)) + ' o mayor')


# "Si se sabe que la calificación de un estudiante es mayor que 72, ¿cuál es la probabilidad de que su calificación sea, de hecho, superior a 84?"
# Para responder a esa pregunta, necesitamos calcular la razón entre el área de la cuva después de 72 con el área después de 84.
# ¿Por que? Porque al calcular el área después de 72 nos estamos restringiendo a la fracción que tiene notas mayores a 72.
# Dentro de ese grupo, queremos saber la probabilidad de que la nota sea mayor a 84. O sea, el ára después de 84 relativo al área después de 72.

# Fraccion de alumnos con calificación mayor a 72:
Area72 = mi_gaussiana.sf(72)
Area84 = mi_gaussiana.sf(82)
# Fracción de alumnos con nota mayor a 72 que tiene nota superior a 84:
print('La proba de que la calificacion sea mayor a 82 sabiendo que es mayor a 72 es de '+ str(round(Area84/Area72, 4)))