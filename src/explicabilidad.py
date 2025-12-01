 # src/explicabilidad.py
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def iniciar_explainer(model, X_train):
    """
    Inicializa el explicador SHAP optimizado para árboles (TreeExplainer)
    pasando solo el modelo para una inicialización más simple.
    """
    print("Inicializando SHAP TreeExplainer (solo modelo)...")
    # Inicializamos solo con el modelo para una configuración base.
    explainer = shap.TreeExplainer(model)
    return explainer

def _limpiar_datos_para_shap(X_data):
    """
    Función de utilidad para limpiar y asegurar tipos numéricos.
    Esta función es la clave para solucionar el ValueError.
    """
    # 1. Forzar a numérico: Intentar convertir cualquier string a float. 
    # 'errors='coerce' convierte lo que no se puede a NaN.
    X_clean = X_data.apply(pd.to_numeric, errors='coerce')
    
    # 2. Rellenar NaN con 0 (o el valor que mejor se adapte a tu imputación)
    X_clean = X_clean.fillna(0)
    
    # 3. Seleccionar solo columnas numéricas y asegurar el tipo float
    X_numeric = X_clean.select_dtypes(include=[np.number]).astype(float)
    return X_numeric

def explicar_global(explainer, X_test):
    """
    Muestra qué características son las más importantes a nivel global usando un Beeswarm plot.
    """
    print("Calculando valores SHAP para el conjunto de prueba (limpiando datos)...")
    
    # Aplicar la limpieza defensiva antes de SHAP
    X_test_numeric = _limpiar_datos_para_shap(X_test)
    
    # Calcular valores SHAP (usamos el método __call__ preferido)
    shap_values_explanation = explainer(X_test_numeric)
    
    plt.figure(figsize=(10, 8))
    plt.title("Impacto Global de las Variables en la Predicción de Fraude")
    
    # Usamos el objeto Explanation completo para el plot
    shap.summary_plot(shap_values_explanation, show=False)
    plt.show() # 

def explicar_prediccion_individual(explainer, X_test, indice_transaccion):
    """
    Explica por qué una transacción específica fue clasificada de una manera,
    usando un Waterfall plot.
    """
    
    # 1. Obtener la instancia específica y limpiarla
    transaccion_df = X_test.iloc[indice_transaccion:indice_transaccion+1]
    transaccion_numeric_df = _limpiar_datos_para_shap(transaccion_df)
    transaccion_numeric_series = transaccion_numeric_df.iloc[0] # Para imprimir
    
    # 2. Calcular el objeto Explanation para esta instancia
    shap_values_single = explainer(transaccion_numeric_df)
    
    print(f"--- Explicando transacción ID {indice_transaccion} ---")
    
    try:
        # F(x) = E[f(x)] + Suma(SHAP values). Calculamos la predicción en log-odds.
        base_value = explainer.expected_value
        total_shap = shap_values_single.values.sum()
        prediccion_log_odds = base_value + total_shap
        print(f"Predicción del modelo (Log-Odds): {prediccion_log_odds:.4f} (Base: {base_value:.4f})")
    except AttributeError:
        print("Predicción Log-Odds no calculada (expected_value no disponible).")
        
    print(f"Valores de la transacción:\n{transaccion_numeric_series.to_string()}")
    
    # Gráfico de cascada (Waterfall plot)
    plt.figure(figsize=(10, 6))
    plt.title(f"Explicación de la Predicción (Log-Odds) para Transacción ID {indice_transaccion}")
    
    # shap_values_single[0] es el objeto Explanation para la única fila
    shap.plots.waterfall(shap_values_single[0], show=False) 
    plt.show() #