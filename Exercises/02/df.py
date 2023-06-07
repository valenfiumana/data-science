# 1. Cargar el archivo como una DataFrame de pandas, asignado a una variable de nombre df.
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt

PATH = "../../Data"  # Path to the dataset directory
FILE = "diamonds.csv"  # Name of the dataset file

def load_data(path=PATH, file=FILE):
    csv_path = os.path.join(path, file)  # Path to the dataset file
    return pd.read_csv(csv_path)  # Read the CSV file using pandas

df = load_data()  # Load the data from the CSV file

# 2. ¿Cuántas filas tiene el dataset? ¿Cuáles son las unidades de este conjunto de datos?
# En otras palabras, ¿de qué habla este dataset?

#print(df)
print('Number of rows: ' + str(len(df. index))) # 53940 filas

# Unidades
# carat: peso en quilates - object
# cut (talla): calidad (Fair, Good, Very Good, Premium, Ideal) - object
# color: de D (mejor) a J (peor) - object
# clarity: claridad del diamante (I1 (peor), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (mejor))
# depth: porcentaje de profundidad total = z / media(x, y)
# table: anchura de la parte superior del diamante en relación con el punto más ancho (43-95)
# precio: USD - int64
# x: longitud en mm
# y: anchura en mm
# z: profundidad en mm

# 3. ¿Cuántas columnas tiene el dataset? ¿Corresponden todas a atributos de las unidades?
print('Number of columns: ' + str(len(df. columns))) # 10 col

# 4. Describir a qué tipo pertenece cada una de las variables.
print(df.info())

# 5. Hay datos faltantes. ¿Cuántos y para qué variables?


# 6. ¿Cuántos tipos de cortes (cut) existen? ¿Cuántos diamantes de cada tipo hay en el dataset?
print('Tipos de cut: '+ str(len(df.cut.unique())))
print(df.groupby('cut').size())

# 7. ¿Cuántos tipos de claridad (clarity) existen? ¿Cuántos diamantes de cada tipo hay en el dataset?
print('Tipos de claridad: '+ str(len(df.clarity.unique())))
print(df.groupby('clarity').size())

# 8. ¿Cómo depende el valor medio del precio del tipo de corte?
mean_price_by_cut = df.groupby('cut')['price'].mean()
print(mean_price_by_cut)
# I. Agrupo datos por el tipo de corte
# II. Calculo el valor medio del precio para cada grupo


# 9. Realizar un gráfico que permita ver esto.
mean_price_by_cut.plot(kind='bar')
plt.xlabel('Tipo de Corte')
plt.ylabel('Valor Medio del Precio')
plt.title('Relación entre el Valor Medio del Precio y el Tipo de Corte')
plt.show()


# 10. Calcular el coeficiente de correlación entre el precio y todas los demás atributos.
# ¿Qué atributo presenta un coeficiente mayor con el precio? Si no conociéramos el precio de los diamantes,
# ¿qué atributo permitiría obtener mayor información sobre el precio?
correlation_matrix = df.corr()
price_correlation = correlation_matrix['price']
print(price_correlation)

# 11. Graficar la matriz de correlación entre todos los parámetros y colorearla.
# Identificar grupos de variables que tienen fuerte correlación o cualquier otro patrón.

# plt.imshow: each cell is represented by a pixel, and the color intensity represents the correlation value
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix)

xt = plt.xticks(np.arange(9), df.columns[:-1], rotation=45, ha='right', va='top')
yt = plt.yticks(np.arange(9), df.columns[:-1], rotation=0, ha='right', va='center')

plt.colorbar(label='Pearson CC')
plt.show()

# sns.heatmap: a color-coded matrix is created where each cell represents the correlation between two parameters
import seaborn as sns
import matplotlib.pyplot as plt

# Obtener la matriz de correlación
corr_matrix = df.corr()

# Crear el gráfico de matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu')

# Configurar los títulos y etiquetas
plt.title('Matriz de Correlación')
plt.xlabel('Parámetros')
plt.ylabel('Parámetros')
# Mostrar el gráfico
plt.show()


