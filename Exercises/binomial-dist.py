# Para acceder a un Master de postgrado se realiza un examen tipo test a los solicitantes.
# El examen consta de cinco preguntas y cada una tiene cuatro posibles resultados.
# Un alumno no conoce las respuestas, pero decide contestar en función del siguiente juego: tira un dado,
# si sale un 1 elige el primer resultado;
# si sale un 2, elige el segundo resultado, y así sucesivamente;
# si sale un 5 ó un 6, tira el dado de nuevo.
# Determinar:
# ¿Cuál es la probabilidad de éxito y cuál la de fracaso?
# ¿Cuál es la probabilidad de que acierte una respuesta?
# ¿Cuál es la probabilidad de acertar dos o tres respuestas?

import numpy as np
import matplotlib.pyplot as plt
import random

def tiradas(N=5):  # N = número de preguntas
    respuestas = []
    # random.seed(10)
    tiro = 0
    while tiro < N:
        dado = random.randint(1, 6)
        # print(dado)
        if (dado < 5):
            respuestas.append(dado)
            tiro += 1
    return respuestas


print(tiradas()) # [4, 3, 3, 2, 3]

# ¿Cuál es la probabilidad de éxito y cuál la de fracaso?
preguntas = tiradas(500000)
p = preguntas.count(1)/len(preguntas)

print("Probabilidad exito p= ",p) # p=  0.250228
print("Probabilidad fracaso q= ",1-p) # q=  0.749772

# ¿Cuál es la probabilidad de que acierte una respuesta?
M = 100000               # número de experimentos
correctas = []
for i in range(M):
  respuestas = tiradas()
  ok = respuestas.count(1)
  correctas.append(ok)

print("Probabilidad que acierte una respuestas es: ",correctas.count(1)/M) # 0.39552

#¿Cuál es la probabilidad de acertar dos o tres respuestas?
print("Probabilidad de acertar 2 o 3 respuestas es:",(correctas.count(2)+correctas.count(3))/M) # 0.35289

# Podemos hacer un gráfico de la distribución binomial usando Scipy
from scipy.stats import binom

n = 100
p = 0.5
x = np.arange(0, n+1)
binomial_pmf = binom.pmf(x, n, p)

print(binomial_pmf)
plt.figure(figsize = (10, 5))
plt.plot(x, binomial_pmf, 'ro')
plt.title(f"Distribución Binomial (n={n}, p={p})")
plt.xlabel('# exito = cara')
plt.ylabel('Probabilidad')
plt.show()



