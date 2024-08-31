import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

# Leer el archivo CSV y almacenar los datos en un DataFrame
df = pd.read_csv('altura_peso.csv') 

# Crear las variables x (altura) y (peso)
x = df['Altura'].values
y = df['Peso'].values

# Crear el modelo de Regresión Lineal usando Keras
model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# Compilar el modelo
model.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error')

# Entrenar el modelo
model.fit(x, y, epochs=100, verbose=1)

# Realizar predicciones
y_pred = model.predict(x)

# Para ver los resultados 👍
plt.scatter(x, y, color='blue', label='Datos reales')
plt.plot(x, y_pred, color='red', label='Predicción')
plt.title('Regresión Lineal entre Altura y Peso')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()
