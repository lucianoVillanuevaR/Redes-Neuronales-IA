"""
Script para inspeccionar el modelo serializado `best_model.joblib`.
Genera un resumen por consola, guarda `best_model_inspect.json` y `confusion_from_joblib.png`.
"""
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Cargando modelo desde best_model.joblib...")
model = joblib.load('best_model.joblib')
print("Modelo cargado.")

# Mostrar parámetros generales
print('\n-- Parámetros del modelo --')
print(model)
try:
    params = model.get_params()
    print('\nget_params():')
    for k in ['hidden_layer_sizes', 'activation', 'alpha', 'learning_rate_init', 'max_iter', 'random_state']:
        if k in params:
            print(f"  {k}: {params[k]}")
except Exception as e:
    print('No se pudieron obtener get_params():', e)

# Mostrar arquitectura y pesos
print('\n-- Arquitectura y pesos --')
if hasattr(model, 'hidden_layer_sizes'):
    print('hidden_layer_sizes =', model.hidden_layer_sizes)
if hasattr(model, 'n_layers_'):
    print('n_layers_ =', model.n_layers_)
if hasattr(model, 'n_iter_'):
    print('n_iter_ =', model.n_iter_)
if hasattr(model, 'loss_curve_'):
    print('loss_curve_ length =', len(model.loss_curve_))
    print('última pérdida =', model.loss_curve_[-1])

# Mostrar shapes de coeficientes
if hasattr(model, 'coefs_'):
    print('\ncoefs_ shapes:')
    for i, c in enumerate(model.coefs_):
        print(f'  layer {i}: {c.shape}')
if hasattr(model, 'intercepts_'):
    print('\nintercepts_ shapes:')
    for i, b in enumerate(model.intercepts_):
        print(f'  layer {i}: {b.shape}')

# Cargar test.csv y evaluar
print('\nCargando test.csv para evaluación...')
try:
    # Usar los mismos parámetros que en el script principal (auto-detectar separador)
    test = pd.read_csv('test.csv', sep=None, engine='python', header=None)
    X_test = test.iloc[:, :-1].values.astype('float32') / 255.0
    y_test = test.iloc[:, -1].values
    print('Datos de test cargados:', X_test.shape)

    print('\nEvaluando modelo en test set...')
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy (desde joblib): {acc:.4f}')

    print('\nReporte de clasificación:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print('\nMatriz de confusión:')
    print(cm)

    # Guardar matriz de confusión como PNG
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor real')
    ax.set_title('Matriz de Confusión - best_model.joblib')
    plt.tight_layout()
    plt.savefig('confusion_from_joblib.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Guardado: confusion_from_joblib.png')

    # Guardar resumen JSON
    summary = {
        'test_accuracy': float(acc),
        'n_iter': int(model.n_iter_) if hasattr(model, 'n_iter_') else None,
        'hidden_layer_sizes': model.hidden_layer_sizes if hasattr(model, 'hidden_layer_sizes') else None,
        'activation': model.activation if hasattr(model, 'activation') else None,
        'loss_last': float(model.loss_curve_[-1]) if hasattr(model, 'loss_curve_') else None
    }
    with open('best_model_inspect.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('✓ Guardado: best_model_inspect.json')

    # Mostrar algunas predicciones de ejemplo
    print('\nAlgunos ejemplos (true -> pred):')
    for i in range(10):
        print(f"  {y_test[i]} -> {y_pred[i]}")

except FileNotFoundError:
    print('ERROR: test.csv no encontrado en el directorio actual.')

print('\nInspección completada.')
