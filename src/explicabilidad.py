import shap
import matplotlib.pyplot as plt
import pandas as pd

# Supongamos que 'model' es tu modelo entrenado (ej. el XGBoost o LightGBM)
# y 'X_test' son tus datos de prueba.

def iniciar_explainer(model, X_train):
    """
    Inicializa el explicador SHAP optimizado para árboles.
    """
    # Para XGBoost, LightGBM y RandomForest usamos TreeExplainer
    explainer = shap.TreeExplainer(model)
    return explainer

def explicar_global(explainer, X_test):
    """
    Muestra qué características son las más importantes a nivel global.
    """
    print("Calculando valores SHAP para el conjunto de prueba...")
    shap_values = explainer.shap_values(X_test)
    
    # Resumen visual (Beeswarm plot)
    # Muestra el impacto de valores altos/bajos en la probabilidad de fraude
    plt.title("Impacto Global de las Variables en la Predicción de Fraude")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.show()

def explicar_prediccion_individual(explainer, X_test, indice_transaccion):
    """
    Explica por qué una transacción específica fue aceptada o rechazada.
    
    Args:
        indice_transaccion (int): Índice en X_test de la operación a explicar.
    """
    # Obtener los datos de esa transacción específica
    transaccion = X_test.iloc[indice_transaccion]
    
    # Calcular valores SHAP para esta instancia
    shap_values_single = explainer(X_test.iloc[indice_transaccion:indice_transaccion+1])
    
    print(f"--- Explicando transacción ID {indice_transaccion} ---")
    print(f"Valores de la transacción:\n{transaccion}")
    
    # Gráfico de cascada (Waterfall plot)
    # Es el MEJOR para explicar decisiones individuales
    # E[f(x)] es la probabilidad base promedio
    # f(x) es la probabilidad predicha para esta transacción
    plt.title(f"¿Por qué se clasificó así la transacción {indice_transaccion}?")
    shap.plots.waterfall(shap_values_single[0]) 
    plt.show()
