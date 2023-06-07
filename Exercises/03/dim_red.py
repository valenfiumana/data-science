import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA

# 1. Carguen los dos csvs en dos dataframes distintos de pandas.
# Agréguenle a cada uno una nueva columna 'SEXO' que tenga los valores 'H' y 'M'
# Luego unan los dos datasets en uno nuevo usando la función de pandas pd.concat([df1, df2]).

# 2. Definan un nuevo dataframe de variables sólo numéricas a partir del anterior, descartando las columnas 'SEXO' y 'SUBJECT_NUMBER'
# (¿tiene sentido quedarse con esta última columna?).
# Luego apliquenle el StandardScaler de sklearn a este nuevo dataframe, y hagan una reducción dimensional usando PCA.
# ¿Con cuántas componentes necesito quedarme para explicar el 95% de la varianza de los datos?

# 3. Ahora hagan otro PCA, pero quedándose sólo con 2 componentes, y hagan un scatterplot de los datos.
# ¿Qué es lo que se ve? Traten de pintar los puntos usando la columna categórica "SEXO" que tiene el dataset original.

# 4. (Opcional). Ahora hagan un PCA con un número reducido de componentes (digamos 8),
# y luego apliquen un TSNE con 2 componentes. Grafiquen los resultados cómo hicieron en el punto anterior.
# ¿Qué se ve ahora? Pueden jugar con el número de componentes del PCA, o sólo hacer TSNE, y ver las diferencias.


# 1
# Cargo los dos datasets en dos Dataframes distintos
ansurMen = pd.read_csv('../../Data/ansurMen.csv')
ansurWomen = pd.read_csv('../../Data/ansurWomen.csv')

# Les agrego la nueva columna de 'SEXO'
ansurMen['SEXO'] = 'H'
ansurWomen['SEXO'] = 'M'

# Ahora armo un único DataFrame combinando los dos anteriores
df = pd.concat([ansurWomen, ansurMen])


# 2
# Armo el dataset numérico
columnas_excluidas = ['SEXO', 'SUBJECT_NUMBER'] # columnas a excluir
new_df = df.drop(columns=columnas_excluidas).select_dtypes(include='number') # primero excluyo columnas, despues digo que solo variables numericas
print(new_df.head())

scaled_data = StandardScaler().fit_transform(new_df)
pca = PCA(n_components=0.95) # if you want to keep 95% of the variance in the original data after applying PCA, you can specify the float 0.95 to the hyperparameter n_components
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# - - - Otra manera de conseguir el 95% de varianza
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
# explained_variance_ratio_ es un atributo de la clase PCA que devuelve la proporción de varianza explicada por cada componente principal.
# np.cumsum() calcula la suma acumulativa de los elementos de un arreglo.

num_components_needed = np.argmax(explained_variance_ratio_cumulative >= 0.95) + 1
# np.argmax() devuelve el índice del primer elemento que cumple una condición dada. En este caso, se utiliza para encontrar el primer índice donde la suma acumulativa supera o iguala el valor de 0.95 (representando el 95% de la varianza explicada).
# Se suma 1 al resultado obtenido de np.argmax() porque los índices en Python comienzan desde 0, y se desea obtener el número de componentes necesarias en lugar del índice.

print("Número de componentes necesarias para explicar el 95% de la varianza:", num_components_needed)


# - - - -  Misma manera que antes + grafico

# Definir qué fracción de la varianza se quiere mantener
var_frac = 0.95

# Calcular la suma cumulativa y hacer su gráfica
cumsum = np.cumsum(pca.explained_variance_ratio_)
# eso nos dice cuanta información es retenida si paramos en cada dimensión

# En qué momento la suma cumulativa llega a var_frac * 100 %?
d = np.argmax(cumsum >= var_frac) + 1
print('Con {} componentes, preservamos el {} de la varianza.'.format(d, var_frac))

plt.figure(figsize=(8,5))
plt.plot(cumsum, linewidth=3)

plt.axvline(d, color="k", ls=":")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")

plt.xlabel("Componentes", fontsize=16)
plt.ylabel("Varianza Explicada", fontsize=16)

plt.grid(True)

# 3
# Hago un PCA a 2d
pca2 = PCA(n_components=2) # n_components: número de componentes con las que nos quedamos
pca2.fit(scaled_data)
pca_data2 = pca2.transform(scaled_data)
# o directamente personas_pca2 = pca2.fit_transform(scaled_data)

# Grafico los resultados
colores = {'H':'g', 'M':'orange'} # Para pintar según el sexo
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
scat = ax.scatter(*pca_data2.T, c=df['SEXO'].map(colores), edgecolors='None', alpha=0.4)
plt.title('PCA 2d')

# Hago una pequeña leyenda manual de los colores
import matplotlib.patches as mpatches
leyenda = []
clase = []
for sexo, color in colores.items():
    clase.append(sexo)
    leyenda.append(mpatches.Rectangle((0,0),1,1,fc=color))
plt.legend(leyenda, clase, loc=4)
plt.show()

# Otro gráfico
# Crear un DataFrame con los datos reducidos a dos componentes
pca_df2 = pd.DataFrame(data=pca_data2, columns=['PC1', 'PC2'])

# Scatter plot de los datos reducidos a dos componentes
plt.scatter(pca_df2['PC1'], pca_df2['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter Plot: Data Reduced to 2 Components')
plt.show()


# 4
# Ahora hago PCA quedándome con 8 componentes
pca_8d = PCA(n_components=8) # n_components: número de componentes con las que nos quedamos
personas_pca8d = pca_8d.fit_transform(scaled_data)

from sklearn.manifold import TSNE

# Hago tsne a 2 dimensiones (tarda un ratito)
tsne = TSNE(n_components=2, random_state=42)
reduced_tsne = tsne.fit_transform(personas_pca8d)

# Grafico los nuevos resultados

# Para pintar según el sexo, defino los siguientes colores
colores = {'H':'g', 'M':'orange'}

# Ahora sí, el scatterplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
scat = ax.scatter(*reduced_tsne.T, c=df['SEXO'].map(colores), edgecolors='None', alpha=0.4)
plt.title('PCA + tSNE 2d')

# Hago una pequeña leyenda manual de los colores
import matplotlib.patches as mpatches

leyenda = []
clase = []
for sexo, color in colores.items():
    clase.append(sexo)
    leyenda.append(mpatches.Rectangle((0,0),1,1,fc=color))
plt.legend(leyenda, clase, loc=4)
plt.show()