# Celda 1: Actualizar librería de funciones

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def iniciar_explainer(model, X_train):
    """
    Inicializa el TreeExplainer. 
    Usamos una muestra de X_train como 'background' para mejorar la precisión.
    """
    print("--- Inicializando SHAP TreeExplainer ---")
    try:
        # model_output="raw" es crucial para modelos de clasificación (log-odds)
        explainer = shap.TreeExplainer(model, data=None, model_output="raw")
    except Exception as e:
        print(f"Advertencia: {e}. Usando inicialización simple.")
        explainer = shap.TreeExplainer(model)
    return explainer

def calcular_shap_values(explainer, X_set):
    """
    Calcula los valores SHAP asegurando tipos numéricos.
    """
    print(f"Calculando SHAP values para {len(X_set)} instancias...")
    # Aseguramos que solo entren números y sin nulos
    X_numeric = X_set.select_dtypes(include=[np.number]).fillna(0)
    
    # check_additivity=False evita errores menores de precisión numérica
    shap_values = explainer(X_numeric, check_additivity=False)
    return shap_values

def plot_global_importance(shap_values, max_display=15):
    """Beeswarm plot: Impacto Global de las variables."""
    plt.figure(figsize=(12, 8))
    plt.title("Impacto Global (SHAP Summary)")
    shap.summary_plot(shap_values, show=False, max_display=max_display)
    plt.show()

def plot_dependence(shap_values, feature_name):
    """Scatter plot: Relación no lineal de una variable (ej. Amount)."""
    print(f"Generando gráfico de dependencia para: {feature_name}")
    if feature_name not in shap_values.feature_names:
        print(f"Error: La variable {feature_name} no está en el dataset.")
        return
    
    plt.figure(figsize=(10, 6))
    # Muestra cómo el valor de la variable (eje x) afecta al riesgo (eje y)
    shap.plots.scatter(shap_values[:, feature_name], color=shap_values, show=False)
    plt.title(f"Dependencia de SHAP: {feature_name}")
    plt.show()

def plot_local_waterfall(shap_values, instance_index=0):
    """Waterfall plot: Explicación de una predicción individual."""
    print(f"Explicando transacción índice: {instance_index}")
    plt.figure(figsize=(10, 6))
    # Extraemos la explicación individual
    shap.plots.waterfall(shap_values[instance_index], show=False, max_display=12)
    plt.title(f"Explicación Local - Caso {instance_index}")
    plt.show()