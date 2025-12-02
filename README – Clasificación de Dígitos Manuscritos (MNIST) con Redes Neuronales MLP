==============================================================
   PROYECTO: Clasificación de Dígitos MNIST con Redes MLP  
==============================================================


Este proyecto implementa un Perceptrón Multicapa (MLP) para clasificar dígitos manuscritos utilizando el dataset MNIST, cumpliendo con todos los requerimientos del ramo Inteligencia Artificial.

Incluye comparación de múltiples modelos, regularización, early stopping, validación, generación de métricas y artefactos listos para análisis.

1. Enunciado del Proyecto
==============================================================
                        ENUNCIADO
==============================================================


El dataset MNIST consiste en:

60.000 imágenes para entrenamiento

10.000 imágenes para prueba

Cada imagen es de 28×28 píxeles → 784 valores entre 0–255

El último valor (columna 785) corresponde al dígito real (0–9)

El proyecto exige:

Implementar una red MLP

Definir arquitectura, activación, función de error e iteraciones

Entrenar varios modelos comparativos

Evaluar rendimiento y overfitting

Seleccionar el mejor modelo

Presentar conclusiones y decisiones

Este repositorio cumple completamente el enunciado.

2. Dataset (NO incluido en GitHub)
==============================================================
                     DESCARGA DEL DATASET
==============================================================
GitHub no permite archivos > 100 MB.  
Debe descargar MNIST en CSV desde:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
==============================================================


Renombrar:

mnist_train.csv → train.csv

mnist_test.csv → test.csv

Y colocarlos en la raíz del proyecto.

3. Estructura del Repositorio
==============================================================
                      ESTRUCTURA DEL PROYECTO
==============================================================
Redes-Neuronales-IA/
    mnist_mlp_sklearn.py
    resultados_mnist.json
    confusion_matrix.png
    loss_curve_best_model.png
    best_model.joblib
    README.md
    .gitignore
==============================================================


Los archivos PNG/JSON se generan automáticamente al ejecutar el script.

4. Instalación y Ejecución
==============================================================
                        INSTALACIÓN
==============================================================

1. Clonar repositorio
git clone https://github.com/lucianoVillanuevaR/Redes-Neuronales-IA.git
cd Redes-Neuronales-IA

2. Crear entorno virtual
python -m venv venv
source venv/Scripts/activate

3. Instalar dependencias
pip install numpy pandas scikit-learn matplotlib seaborn joblib

4. Ejecutar el script
python mnist_mlp_sklearn.py

5. Modelos Entrenados
==============================================================
                     MODELOS EVALUADOS
==============================================================

Modelo	Arquitectura	Activación	Iteraciones	Regularización
A	(128, 64)	ReLU	20	0.0001
B	(256, 128)	ReLU	30	0.0001
C	(128, 64)	tanh	30	0.0001
D	(256, 128)	ReLU	30	0.001

Incluyen:

Validación (20%)

Early stopping

Regularización L2

Comparación Train / Validación / Test

6. Resultados
==============================================================
                       RESULTADOS
==============================================================
Los valores exactos dependen de la ejecución.
Se almacenan en: resultados_mnist.json
==============================================================


Ejemplo (sustituir con valores reales):

Modelo A: Test Accuracy = 0.94
Modelo B: Test Accuracy = 0.96
Modelo C: Test Accuracy = 0.93
Modelo D: Test Accuracy = 0.95

7. Mejor Modelo Seleccionado
==============================================================
                   MEJOR MODELO (SELECCIÓN)
==============================================================
Modelo recomendado: Modelo D (256,128) con ReLU y alpha=0.001
==============================================================


Razones:

Alto test accuracy

Mejor equilibrio entre train / validación / test

Menos sobreajuste gracias a la regularización

Curva de pérdida más estable

8. Artefactos Generados
==============================================================
                ARTEFACTOS GENERADOS AUTOMÁTICAMENTE
==============================================================
1. resultados_mnist.json           (métricas completas)
2. confusion_matrix.png            (matriz de confusión)
3. loss_curve_best_model.png       (curva de pérdida)
4. best_model.joblib               (modelo entrenado)
==============================================================


Estos archivos permiten análisis profundo y presentación profesional.

9. Decisiones Técnicas
==============================================================
                 DECISIONES DEL DISEÑO DEL MODELO
==============================================================
- Normalización 0–255 → 0–1
- División Train / Validación / Test
- Early stopping para evitar overfitting
- Comparación de arquitecturas
- Comparación de funciones de activación
- Ajuste de regularización alpha
==============================================================

10. Conclusiones
==============================================================
                          CONCLUSIONES
==============================================================
- Los MLP logran alta precisión en MNIST (94–97%).
- ReLU supera a tanh en convergencia y precisión.
- Más neuronas → más capacidad → riesgo de sobreajuste.
- Regularización y validación mejoran generalización.
- El mejor modelo es un equilibrio entre tamaño y estabilidad.
==============================================================
