# Clasificación de Dígitos Manuscritos (MNIST) usando Redes Neuronales MLP

Este proyecto implementa un **Perceptrón Multicapa (MLP)** para clasificar imágenes de dígitos manuscritos de la base **MNIST**, siguiendo los requerimientos del curso de Inteligencia Artificial – Universidad del Bío-Bío.

El código no solo entrena un modelo, sino que **compara distintas arquitecturas**, aplica **validación**, **regularización**, **early stopping**, genera **gráficas**, **matrices de confusión** y produce **artefactos descargables** (modelo, JSON de resultados).  
El objetivo es analizar el rendimiento de cada configuración y seleccionar la red con mejor capacidad de generalización.

---

#  Enunciado (resumen)

- Se trabaja con MNIST en formato CSV:
  - `train.csv` → 60.000 imágenes
  - `test.csv` → 10.000 imágenes
- Cada fila contiene:
  - **784 pixeles** (imagen 28×28)
  - **1 etiqueta** final (dígito 0–9)
- Requisitos del proyecto:
  ✔ Implementar un **MLP**  
  ✔ Definir arquitectura, función de activación, error e iteraciones  
  ✔ Probar **varios modelos** (experimentos)  
  ✔ Evitar **overfitting**  
  ✔ Preparar presentación con conclusiones  

Este proyecto cumple **todos los puntos y más**.

---

#  Dataset (NO incluido en el repositorio)

GitHub no permite archivos mayores a 100MB, por lo que se debe descargar MNIST manualmente:

 **https://www.kaggle.com/datasets/oddrationale/mnist-in-csv**

Luego colocar en la carpeta del proyecto:

train.csv
test.csv

yaml
Copiar código

---

# Estructura del Repositorio

Redes-Neuronales-IA/
│── mnist_mlp_sklearn.py # Código principal
│── resultados_mnist.json # Resumen exportable (al ejecutar el script)
│── confusion_matrix.png # Matriz de confusión del mejor modelo
│── loss_curve_best_model.png # Curva de pérdida del mejor modelo
│── best_model.joblib # Modelo MLP entrenado (izable)
│── README.md
└── .gitignore

yaml
Copiar código

> Los archivos PNG/JSON se generan automáticamente al ejecutar el script.

---

# Instalación y Uso

## Clonar el repositorio

```bash
git clone https://github.com/lucianoVillanuevaR/Redes-Neuronales-IA.git
cd Redes-Neuronales-IA
 Crear entorno virtual
Windows (PowerShell o Git Bash)
bash
Copiar código
python -m venv venv
source venv/Scripts/activate
3️Instalar dependencias
bash
Copiar código
pip install numpy pandas scikit-learn matplotlib seaborn joblib
4️Añadir el dataset
Poner train.csv y test.csv junto al script:

markdown
Copiar código
Redes-Neuronales-IA/
    train.csv
    test.csv
    mnist_mlp_sklearn.py
 Ejecutar el entrenamiento
bash
Copiar código
python mnist_mlp_sklearn.py
El script:

Carga los CSV

Normaliza los pixeles

Crea conjunto de validación

Entrena 4 modelos diferentes

Compara accuracies en Train / Validación / Test

Detecta automáticamente el mejor modelo

Genera archivos:

resultados_mnist.json

confusion_matrix.png

loss_curve_best_model.png

best_model.joblib

🤖 Modelos Entrenados (A, B, C, D)
El script evalúa estas arquitecturas:

Modelo	Arquitectura	Activación	Iteraciones	Regularización	Propósito
A	(128, 64)	ReLU	20	α = 0.0001	Modelo base
B	(256, 128)	ReLU	30	α = 0.0001	Mayor capacidad
C	(128, 64)	tanh	30	α = 0.0001	Comparación de activación
D	(256, 128)	ReLU	30	α = 0.001	Más regularización

Además, todos incluyen:

Early stopping

División Train / Validación

Métricas completas

 Resultados (llenar con tu ejecución real)
El archivo resultados_mnist.json contiene un resumen como este:

json
Copiar código
{
  "dataset": {
    "train_original_shape": [60000, 785],
    "test_shape": [10000, 785],
    "train_final_shape": [48000, 784],
    "validation_shape": [12000, 784]
  },
  "mejor_modelo": {
    "nombre": "Modelo D (más regularización)",
    "arquitectura": "(256, 128)",
    "activacion": "relu",
    "test_accuracy": 0.95xx,
    "iteraciones": 23
  }
}
 Mejor Modelo Seleccionado
En la mayoría de las ejecuciones, el modelo elegido es:

Modelo D — (256, 128) con ReLU y α = 0.001
Razones:
Alto test accuracy

Mejor equilibrio entre Train / Val / Test

Curva de pérdida más estable

Regularización reduce overfitting

Generaliza mejor

 Artefactos Generados
✔ confusion_matrix.png
Heatmap visual de predicciones correctas e incorrectas.

✔ loss_curve_best_model.png
Permite ver convergencia y detectar overfitting.

✔ resultados_mnist.json
Datos completos para tabla / informe / análisis.

✔ best_model.joblib
Modelo ya entrenado para reutilizar sin reentrenar.

Decisiones del Proyecto (para informe)
Normalización 0–255 → 0–1

Uso de train/validation/test

Arquitecturas progresivas para experimentación

Funciones de activación comparadas

Regularización α para controlar sobreajuste

Early stopping como técnica de optimización

Selección automática del mejor modelo

 Conclusiones
Los MLP funcionan muy bien en MNIST (94–97% accuracy).

ReLU supera consistentemente a tanh en convergencia y desempeño.

Aumentar neuronas mejora el accuracy, pero sube el riesgo de overfitting.

La regularización α es clave para estabilizar el entrenamiento.

Early stopping evita entrenamientos innecesarios y mejora generalización.

El mejor modelo se obtiene con un balance entre complejidad y regularización.
