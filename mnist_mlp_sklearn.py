print("=== INICIO SCRIPT MLP ===")

import time
start_time = time.time()

print("Importando librerías...")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

print("Librerías importadas correctamente.")
print("Directorio actual:", os.getcwd())

# 1. Cargar datos desde CSV
print("Cargando train.csv y test.csv...")

try:
    # sep=None + engine="python" deja que pandas detecte el separador (coma, punto y coma, tab, etc.)
    train = pd.read_csv("train.csv", sep=None, engine="python", header=None)
    test = pd.read_csv("test.csv", sep=None, engine="python", header=None)
except FileNotFoundError as e:
    print("ERROR: No se encontró algún CSV.")
    print("Detalle:", e)
    print("Asegúrate de que train.csv y test.csv estén en esta carpeta:")
    print(os.getcwd())
    raise SystemExit

print("Datos cargados. Tamaño Train:", train.shape, "Tamaño Test:", test.shape)
print("Primeros 10 valores de la primera fila de train:")
print(train.iloc[0, :10])

# Suponemos que la última columna es la etiqueta (dígito 0-9)
X_train = train.iloc[:, :-1].values  # todas menos la última -> 784 pixeles
y_train = train.iloc[:, -1].values   # última columna -> etiqueta

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_test:", y_test.shape)

# 2. Normalizar pixeles 0-255 -> 0-1
print("Normalizando pixeles...")
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# ============================================================================
# Crear conjunto de validación (para evitar overfitting y usar early stopping)
# ============================================================================
print("Creando conjunto de validación (80% train / 20% validación sobre el train)...")
# Usamos stratify para mantener la proporción de clases
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"✓ Train final: {X_train.shape}, Validación: {X_val.shape}")

# Mostrar claramente las decisiones requeridas por el enunciado:
print("\nDECISIONES (a declarar en el informe):")
print("- Arquitecturas evaluadas: varias (ej. (128,64), (256,128), (128,64) tanh, (256,128) alpha alto)")
print("- Funciones de activación probadas: ReLU, tanh")
print("- Función de error: Cross-Entropy (implícita en MLPClassifier para clasificación multiclase)")
print("- Iteraciones máximas (por modelo): definidas en cada configuración (max_iter)")


# 3. Definir distintos modelos a probar
modelos = [
    {
        "nombre": "Modelo A (base)",
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "max_iter": 20,
        "alpha": 0.0001
    },
    {
        "nombre": "Modelo B (más neuronas)",
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "max_iter": 30,
        "alpha": 0.0001
    },
    {
        "nombre": "Modelo C (otra activación)",
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "max_iter": 30,
        "alpha": 0.0001
    },
    {
        "nombre": "Modelo D (más regularización)",
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "max_iter": 30,
        "alpha": 0.001
    },
]

resultados = []

# 4. Entrenar cada modelo y guardar resultados
for cfg in modelos:
    print("\n==============================")
    print("Entrenando", cfg["nombre"])
    print("Arquitectura:", cfg["hidden_layer_sizes"],
          "| activación:", cfg["activation"],
          "| max_iter:", cfg["max_iter"],
          "| alpha:", cfg["alpha"])
    print("==============================")

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation=cfg["activation"],
        solver='adam',
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=cfg["max_iter"],
        alpha=cfg["alpha"],
        # Para ayudar a evitar overfitting usamos early stopping
        early_stopping=True,
        validation_fraction=0.1,  # 10% del train usado internamente para early stopping
        n_iter_no_change=10,
        verbose=True,
        random_state=42
    )

    mlp.fit(X_train, y_train)
    # Evaluar en train/val/test
    train_accuracy = mlp.score(X_train, y_train)
    val_accuracy = mlp.score(X_val, y_val)
    test_accuracy = mlp.score(X_test, y_test)

    print(f"{cfg['nombre']} - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
    print(f"Iteraciones realizadas: {mlp.n_iter_}")

    resultados.append({
        "nombre": cfg["nombre"],
        "arquitectura": str(cfg["hidden_layer_sizes"]),
        "activacion": cfg["activation"],
        "max_iter": cfg["max_iter"],
        "alpha": cfg["alpha"],
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "iteraciones": mlp.n_iter_,
        "modelo": mlp
    })

# 5. Mostrar resumen final de todos los modelos
print("\n=== RESUMEN DE MODELOS ===")
for r in resultados:
    print(
        f"{r['nombre']}: "
        f"arch={r['arquitectura']}, act={r['activacion']}, "
        f"max_iter={r['max_iter']}, alpha={r['alpha']}, "
        f"test_acc={r['test_accuracy']:.4f}, iters={r['iteraciones']}"
    )

# 6. Opcional: mostrar matriz de confusión y curva de pérdida del MEJOR modelo
#    (el de mayor accuracy en test)
mejor = max(resultados, key=lambda x: x["test_accuracy"])
print("\n=== MEJOR MODELO ===")
print(
    f"{mejor['nombre']}: arch={mejor['arquitectura']}, act={mejor['activacion']}, "
    f"max_iter={mejor['max_iter']}, alpha={mejor['alpha']}, "
    f"test_acc={mejor['test_accuracy']:.4f}, iters={mejor['iteraciones']}"
)

mejor_mlp = mejor["modelo"]
y_pred = mejor_mlp.predict(X_test)

print("\nMatriz de confusión del mejor modelo:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nReporte de clasificación del mejor modelo:")
print(classification_report(y_test, y_pred))

if hasattr(mejor_mlp, "loss_curve_"):
    plt.figure()
    plt.plot(mejor_mlp.loss_curve_)
    plt.xlabel("Iteración")
    plt.ylabel("Loss")
    plt.title(f"Curva de pérdida - {mejor['nombre']}")
    plt.grid(True)
    plt.show()

end_time = time.time()
print(f"\nTiempo total de ejecución: {end_time - start_time:.2f} segundos")
print("=== FIN SCRIPT MLP ===")
