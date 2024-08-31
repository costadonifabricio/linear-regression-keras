import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

## Lee el archivo CSV y devuelve las variables x (altura) y y (peso).
def cargar_datos(ruta_csv):
    df = pd.read_csv(ruta_csv)
    x = df['Altura'].values
    y = df['Peso'].values
    return x, y

## Crea y compila un modelo de Regresión Lineal usando Keras.
def crear_modelo():
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='linear'))
    model.compile(optimizer=SGD(learning_rate=0.0004), loss='mean_squared_error')
    return model

## Entrena el modelo con los datos de altura (x) y peso (y).
def entrenar_modelo(model, x, y, epochs=100):
    model.fit(x, y, epochs=epochs, verbose=1)
    return model

## Realiza predicciones con el modelo entrenado.
def realizar_predicciones(model, x):
    return model.predict(x)

## Muestra los resultados de la predicción comparados con los datos reales.
def visualizar_resultados(x, y, y_pred):

    plt.scatter(x, y, color='blue', label='Datos reales')
    plt.plot(x, y_pred, color='red', label='Predicción')
    plt.title('Regresión Lineal entre Altura y Peso')
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.legend()
    plt.show()

# Ejecución del código
ruta_csv = 'altura_peso.csv'
x, y = cargar_datos(ruta_csv)
model = crear_modelo()
model = entrenar_modelo(model, x, y)
y_prediccion = realizar_predicciones(model, x)
visualizar_resultados(x, y, y_prediccion)
