import pandas as pd
from matplotlib import pyplot as plt

# 1. Cargando los datos. Importen este nuevo dataset usando pandas.
# Van a notar que les da una advertencia (warning) porque hay algunas columnas con tipos mezclados. Por ahora ignorenlo.
df = pd.read_csv('../../Data/arbolado-publico-lineal-2017-2018.csv')
df = df[['nro_registro', 'nombre_cientifico', 'estado_plantera', 'ubicacion_plantera', 'nivel_plantera', 'diametro_altura_pecho', 'altura_arbol']].copy()
# o sino cols = ['nro_registro', 'nombre_cientifico', 'estado_plantera', 'ubicacion_plantera', 'nivel_plantera', 'diametro_altura_pecho', 'altura_arbol']
# df = df[cols]
print(df.head())

# 2. Limpieza de datos (I). Analicen los valores únicos que pueden tomar las columnas 'estado_plantera', 'ubicacion_plantera' y 'nivel_plantera'.
# ¿Qué es lo que ven?
# Para las tres columnas, unifiquen los valores que pertecen a una misma catgoría.

print('Estado:')
print(df['estado_plantera'].unique())
print('Ubicacion:')
print(df['ubicacion_plantera'].unique())
print('Nivel: ')
print(df['nivel_plantera'].unique())

from sklearn import preprocessing

# Hay muchas categorías que son lo mismo, pero están escrito distinto. Para
# corregirlo, hagamos unos diccionarios que nos indiquén por qué valor habría
# que reemplazarlos:

# lowercase
df['estado_plantera'] = df['estado_plantera'].str.lower()
df['ubicacion_plantera'] = df['ubicacion_plantera'].str.lower()
df['nivel_plantera'] = df['nivel_plantera'].str.lower()

# UBICACION
dic_ubicacion = {'regular ':'regular',
            'och':'ochava', 'ochva':'ochava',
            'fuera línea,Ochava':'ochava/fuera línea',
            'fuera de línea, Ochava':'ochava/fuera línea',
            'fuera línea/ochava':'ochava/fuera línea'}
# reemplazo:
for clave, valor in dic_ubicacion.items():
    df.ubicacion_plantera.replace(clave, valor, inplace=True)


# NIVEL
dic_nivel = {'a  nivel':'a nivel', 'a nivel':'a nivel', 'a nivel':'a nivel',
            'a nivel ':'a nivel', 'an':'a nivel', 'an':'a nivel',
            'baja nivel':'bajo nivel', 'bajo  nivel':'bajo nivel',
            'bajo bivel':'bajo nivel', 'bajo nivel':'bajo nivel',
            'bajo nivel':'bajo nivel', 'bn':'bajo nivel', 'el':'elevada',
            'el':'elevada', 'elevadas':'elevada', 'elevado':'elevada', 'eleveda':'elevada'}
# reemplazo:
for clave, valor in dic_nivel.items():
    df.nivel_plantera.replace(clave, valor, inplace=True)

print('Estado:')
print(df['estado_plantera'].unique())
print('Ubicacion:')
print(df['ubicacion_plantera'].unique())
print('Nivel: ')
print(df['nivel_plantera'].unique())

# 3. Limpieza de datos (II). Hagan histogramas de los valores de las variables 'diametro_altura_pecho' y 'altura_arbol'.
# A primera vista no parece haber nada raro, pero fijense que para el diámetro (que está medido en cm) hay muchos datos con valor 0 (pueden usar el método value_counts()).
# Si bien podría haber árboles con menos de 1 cm de diámetro, la cantidad de los mismos nos hace sospechar que en gran parte de los casos se trata de un error.
# Eliminen las filas con diámetro 0, o al menos por ahora reemplacen el valor por nan.

import numpy as np
columns = ['diametro_altura_pecho', 'altura_arbol']

N_col = 2
N_rows = 1

fig, ax = plt.subplots(N_rows,N_col, figsize=(5*N_col,5*N_rows)) # he resulting figure object is stored in the variable fig, and an array of subplot axes objects is stored in the variable ax
# The ax variable is an array of size (N_rows, N_col), representing the grid of subplots.
# Each element in the ax array corresponds to an individual subplot
# For example, if you have a grid of 1 rows and 2 columns, ax[0] refers to the subplot axes in the first row and first column, ax[1] refers to the subplot axes in the first row and second column, and so on.


for i in range(N_col):
        ax[i].hist(df[columns[i*N_rows]], bins=70) # plots a histogram for each column of the df
        ax[i].set_title(columns[i*N_rows])
# When i is 0: i*N_rows will be 0 * 2 = 0 --> This means the first subplot (index 0) will plot the column at index 0.
# When i is 1: i*N_rows will be 1 * 2 = 2 --> This means the second subplot (index 1) will plot the column at index 2.
# Therefore, the resulting figure will have two subplots arranged horizontally.

plt.show()

# Veo cuantos arboles con 0 hay:
print('N arboles diam 0:', df.diametro_altura_pecho.value_counts()[0.])
print('N arboles diam 1:', df.diametro_altura_pecho.value_counts()[1.])

# Por las dudas reemplacemos por Nan:
df.diametro_altura_pecho.replace(0., np.nan, inplace=True)

# 4. Datos faltantes. Analicen la cantidad de datos faltantes en cada columna y decidan qué hacer con ellos
# (descartarlos, crear una nueva categoría en las variables categóricas, reemplazarla por promedio/mediana en las numéricas, etc.)

# Veamos un resumen de los datos faltantes:
df.isna().sum()

# - - - - BORRAR: La verdad es que con 370000 datos que tiene el dataset, perder un pocos miles no es un problema:
df_limpio = df.dropna()

# - - - - IMPUTAR: Pero si quisiésemos imputar los datos numéricos, lo hacemos de la siguiente manera:
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median') # Uso la mediana

# Para usar este SimpleImputer, tenemos que dejar de lado las variables categóricas, ya que sólo funciona con variables numéricas.
# ADVERTENCIA: La columna 'nro_registro' esta importada como string porque hay algún error, así que la tenemos que sacar para que funcione:
columnas_categoricas = ['nombre_cientifico', 'estado_plantera',
                        'ubicacion_plantera', 'nivel_plantera', 'nro_registro']
df_limpio_num = imputer.fit_transform(df.drop(columnas_categoricas, axis=1))

# Para las variables categóricas, podemos crear una nueva categoría que sea "sin datos" (S/D), o "datos faltantes":
columnas_categoricas = ['nombre_cientifico', 'estado_plantera', 'ubicacion_plantera', 'nivel_plantera']
df_limpio_cat = df[columnas_categoricas].fillna('S/D')
df_limpio_cat.ubicacion_plantera.value_counts()

# 5. Variables categóricas. Apliquen el método de One-Hot Encoding a alguna de las variables categóricas del dataset.
# ¿De qué va a depender la cantidad de componentes de los vectores resultantes?
# Usemos el OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

# Ajustamos a la data categorica de ubicacion_plantera
# OJO que el OneHotEncoder espera recibir un dataframe, no una sola columna, por eso el doble paréntesis abajo
cat_encoder.fit(df_limpio_cat[['ubicacion_plantera']])

#lo usamos para transformar los datos categoricos.
#si bien se "ajusto" con la data de entrenamiento, lo usamos para transformar los datos de Test
ubicacion_plantera_OHE = cat_encoder.transform(df_limpio_cat[['ubicacion_plantera']]).toarray()

# Veamos cómo se ven después de codificar las categorías
for categorical_value in df_limpio_cat['ubicacion_plantera'].unique():
    print(f'{categorical_value:<14s}' + "---> \t" + str(cat_encoder.transform([[categorical_value]]).toarray()[0]))