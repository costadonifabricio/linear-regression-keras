import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

## Lee el archivo CSV y devuelve las variables x (altura) y y (peso).
def cargar_datos(ruta_csv):
    datos = pd.read_csv(ruta_csv, sep=',', header=0)
    x = datos["Altura"].values
    y = datos["Peso"].values
    return x, y

## Crea y compila un modelo de Regresión Lineal usando Keras.
def crear_modelo(input_dim, output_dim, learning_rate):
    np.random.seed(2)
    modelo = Sequential()
    modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))
    sgd = SGD(learning_rate=learning_rate)
    modelo.compile(loss='mse', optimizer=sgd)
    modelo.summary()
    return modelo

## Entrena el modelo con los datos de altura (x) y peso (y).
def entrenar_modelo(modelo, x, y, num_epochs, batch_size):
    historia = modelo.fit(x=x, y=y, epochs=num_epochs, batch_size=batch_size, verbose=1)
    return historia

## Muestra los resultados de la predicción comparados con los datos reales.
def visualizar_resultados(historia, x, y, y_regr):
    plt.subplot(1, 2, 1)
    plt.plot(historia.history['loss'])
    plt.xlabel('epoch')
    plt.ylabel('ECM')
    plt.title('ECM vs. epochs')

    plt.subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.plot(x, y_regr, color='red', label='linea de predicción')
    plt.title('Datos originales y regresión lineal')
    plt.show()

## Realiza una predicción para una altura específica.
def realizar_prediccion(modelo, x_pred):
    y_pred = modelo.predict(x_pred)
    return y_pred

# Ejecución del código
ruta_csv = 'altura_peso.csv'
x, y = cargar_datos(ruta_csv)
modelo = crear_modelo(input_dim=1, output_dim=1, learning_rate=0.00003)
historia = entrenar_modelo(modelo, x, y, num_epochs=100, batch_size=len(x))

capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0], b[0]))

y_regr = modelo.predict(x)
print('ECM = {:.2f}'.format(np.mean((y_regr - y) ** 2)))
visualizar_resultados(historia, x, y, y_regr)

x_pred = np.array([170])
y_pred = realizar_prediccion(modelo, x_pred)
print(f"Peso predicho: {y_pred[0][0]:.2f} kg")