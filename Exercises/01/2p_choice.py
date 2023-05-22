# En el examen del final del módulo de Python, los participantes tenían que responder a 10 preguntas multiple-choice
# con cuatro opciones, siendo solo una de ellas la correcta.
# a. Simular el experimento de responder a las preguntas aleatoriamente y guardar la cantidad de respuestas correctas en cada caso.
# Hacer un histograma con los resultados obtenidos.
# b. Sabiendo que se aprueba con al menos 6 respuestas correctas, ¿Cuál es la probabilidad de aprobar respondiendo aleatoriamente según el experimento anterior?
# c. Calcular el valor medio, la mediana y la moda de los resultados obtenidos.

import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# FORMA 1: Supongo que 1 es la rta correcta
# - - - - - - - - - - - - - - - - - - - - - -

N = 100000  # Simulo 100.000 exámenes, c/u de 10 preguntas
results = []

for i in range(N):
    rightAnswers = 0
    #print('Exam '+ str(i))
    for i in range(10):
        res = random.randint(1, 4)  # number between 1 and 4
        #print(res)
        if res == 1: rightAnswers += 1  # Here I chose that the correct answer is 1, but could be 2, 3 or 4

    results.append(rightAnswers)
    #print('Right answers: '+ str(rightAnswers))

print(np.mean(results))


# FORMA 2: Genero una respuesta correcta random por iteración
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


N = 100000  # Simulate 100,000 exams, each with 10 questions
results = []

answer_choices = [1, 2, 3, 4]  # Available answer choices

for i in range(N):
    rightAnswers = 0
    for i in range(10):
        correct_answer = random.choice(answer_choices)  # Randomly select the correct answer
        #print('Correct answer: ' + str(correct_answer))
        res = random.choice(answer_choices)  # Randomly select an answer choice
        #print('Actual answer: ' + str(res))
        if res == correct_answer:
            rightAnswers += 1

    results.append(rightAnswers)
    #print('Right answers: '+ str(rightAnswers))

print(np.mean(results))

# HISTOGRAMA
# - - - - - - - - - - - - - - - - - - - - - -
plt.figure()
data = np.random.randn(1000)
num_bins = int(1 + np.log2(len(data)))
plt.hist(results, bins=num_bins)
plt.xlabel('calificaciones')
plt.ylabel('cantidad')
plt.show()

# CALCULAR PROBA APROBAR
# - - - - - - - - - - - - - - - - - - - - - -
# Forma larga
sum = 0
for i in results:
    if(i>=6):
        sum+=1
print('La probabilidad de tener 6 o mas respuestas correctas es de ' + str(sum/N))

# Forma corta
passedExams = len([num for num in results if num >= 6])
print('La probabilidad de tener 6 o mas respuestas correctas es de ' + str(passedExams/N))
# [num for num in results if num > 6] filters out the numbers that are greater than 6
# len() counts the number of elements in the filtered list (the number of exams that were passed).

# MEAN, MEDIAN, MODE
# - - - - - - - - - - - - - - - - - - - - - -
print('Promedio:', np.mean(results))
print('Mediana:', np.median(results))
print('Moda:', stats.mode(results).mode)