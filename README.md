# Regresión Lineal con Keras

Este proyecto realiza una regresión lineal simple utilizando la biblioteca Keras para predecir el peso de una persona en función de su altura. Los datos se cargan desde un archivo CSV y se utiliza un modelo de red neuronal simple para entrenar y realizar predicciones.


## Entorno Virtual

Cree un entorno virtual con el siguiente comando:

```bash
virtualenv nombre del entorno
```

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas antes de ejecutar el código:

- Python
- pandas
- numpy
- matplotlib
- tensorflow
- keras

Puedes instalar las dependencias necesarias utilizando pip:

```bash
pip install pandas numpy matplotlib tensorflow keras
```

## Visualización de Resultados:
El modelo arrojaba los siguientes resultados con learning_rate a 0.0004:
![alt text](<Captura de pantalla 2024-09-03 093657.png>)

El modelo arrojaba los siguientes resultados con leraning_rate a 0.00003:
El ECM se calcula para evaluar la precisión del modele.
Datos Originales y Regresión Lineal, compara los datos reales con la línea de regresión predicha. El resultado muestra el peso predicho para una persona con una altura de 170 cm.
![alt text](<Captura de pantalla 2024-09-03 093818.png>)
