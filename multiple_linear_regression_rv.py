# Regresión múltiple sobre todos los datos - Predicción de satisfacción de empleados

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Cargar el dataset desde CSV
dataset = pd.read_csv('Employee_Attrition.csv')

# 2. Limpiar datos (rellenar o eliminar valores faltantes)
# Rellenamos numéricos con la media
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

# Rellenamos categóricos con el valor más frecuente
for col in ['dept', 'salary']:
    if dataset[col].isnull().sum() > 0:
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

# 3. Definir variables independientes y dependiente
variables_numericas = ["last_evaluation","number_project","average_montly_hours",
                       "time_spend_company","Work_accident","promotion_last_5years"]
variables_categoricas = ["dept","salary"]
X = dataset[variables_numericas + variables_categoricas]
y = dataset['satisfaction_level'].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# 4. Codificar variables categóricas usando OneHotEncoder
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), ['dept', 'salary'])
], remainder='passthrough')

# Aplicamos la transformación y convertimos X en un array de NumPy
X = np.array(ct.fit_transform(X))

# 5. División del dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
# 6. Entrenamiento del modelo de regresión lineal múltiple
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# 7. Predicción de los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)

# 8. Comparar predicciones con valores reales
np.set_printoptions(precision=2)
resultados = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)
print("Predicciones vs Valores Reales:\n", resultados)

# Error cuadratico medio: Mide el promedio de los errores al cuadrado significa Cuanto más bajo, mejor el modelo
from sklearn.metrics import mean_squared_error, r2_score
error_cuadratico_medio = mean_squared_error(y_test, y_pred)

# RCuadrado Que tan buena llega a ser una prediccion
r_cuadrado = r2_score(y_test,y_pred)
print(f"Error cuadrático medio (MSE): {error_cuadratico_medio:.4f}")
print(f"Coeficiente de determinación (R²): {r_cuadrado:.4f}")

# 9. Visualización de los resultados
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores reales')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicciones')
plt.title('Predicciones vs Valores reales - Satisfacción del empleado')
plt.xlabel('Índice de muestra')
plt.ylabel('Nivel de satisfacción')
plt.legend()
plt.grid()
plt.show()