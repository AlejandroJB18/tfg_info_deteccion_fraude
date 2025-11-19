# tfg_info_deteccion_fraude
Desarrollar un sistema de Machine Learning que no solo clasifique transacciones como fraudulentas o legítimas, sino que **minimice la pérdida financiera total esperada**. El modelo debe aprender que un fraude de 10,000€ es significativamente más costoso que 10 fraudes de 100€.

### Enfoque Técnico
* **Problema Imbalanceado:** Tratamiento de la baja tasa de fraude (clase minoritaria).
* **Sensibilidad al Coste:** Incorporación del campo `Amount` (monto) como factor de penalización durante el entrenamiento y la evaluación.
* **Métrica Clave:** **Costo Financiero Esperado** (Expected Financial Cost), en lugar de F1-Score o Balance Accuracy.

---

## 📂 Estructura y Código

El código se diseña para una ejecución  **secuencial**, adecuada para entornos de desarrollo y depuración como **Visual Studio** o **Spyder**. El código se ha organizado para separar la lógica reutilizable (`src/`) de los experimentos y análisis finales (`notebooks/`).

```plaintext
/data/                                  → Datasets (credit_card.csv, cs_training, etc.)
/exploracion/                           → Notebooks de la fase inicial de investigación
    ├── deteccion_impago...ipynb      
    ├── varios_datasets.ipynb     
    ├── modelos_avanzados.ipynb      
/notebooks/                             → Análisis finales
    ├── analisis_financiero.ipynb       # Simulación y optimización del modelo
    ├── analisis_sensibilidad.ipynb     # Análisis de robustez de negocio
/results/                               → Modelos, métricas y gráficos
/src/                                   → Módulos Python
    ├── load_data.py                    # Carga y limpieza de datos
    ├── train_model.py                  # Entrenamiento con coste variable
    ├── evaluate.py                     # Función de Coste Financiero y Optimización de Umbral
    └── compare_models.py               # Simulaciones
README.md                               → Este documento
```
### Ejemplos de código

```python
# 1. Cargar datos y separar importes
from src.load_data import load_fraud_csv
df, X, y = load_fraud_csv('data/credit_card.csv')

# 2. Entrenar modelo con penalización variable (Amount * factor)
from src.train_model import train_xgb_with_cost
# amount_factor=20 indica que el modelo debe priorizar 20 veces más el importe
xgb = train_xgb_with_cost(X_train, y_train, amount_train, amount_factor=20)

# 3. Encontrar el umbral que minimiza el coste real
from src.evaluate import best_threshold_by_cost
# Coste = (FN * 90% del Importe) + (FP * 5€ inspección)
best_thr, min_cost = best_threshold_by_cost(y_test, proba, amount_test)
```
Dataset Credit Card Fraud Detection de Kaggle [data/creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
## Modelos Comparados

| Modelo         | Ventajas                                  | Desventajas                              |
|----------------|-------------------------------------------|------------------------------------------|
| **RandomForest** | Interpretable, robusto a outliers         | Menos sensible a pesos complejos         |
| **XGBoost**      | Mejor rendimiento en coste, optimización por gradiente | Menos interpretable, requiere más tuning |

> **Resultado típico:** XGBoost reduce el coste esperado en **~15-25%** vs. RandomForest.

---

## Métricas Clave

- **Expected Financial Cost** = `(FN × Amount × 0.9) + (FP × 5)`
- **AUPRC** (para comparación)
- **Mejor umbral** optimizado por coste, no por F1.

---

## 📈 Ejemplo de Resultados del código *fraud_detection.py*

```python
from src.compare_models import compare_all_factors
compare_all_factors('data/creditcard.csv')
```
## 📊 Resultados Reales (04-Nov-2025)

| amount_factor | Modelo   | Umbral | Coste (€) | AUPRC |
|---------------|----------|--------|-----------|-------|
| 5             | XGBoost  | 0.2575 | 1 693     | 0.8838 |
| 10            | XGBoost  | 0.3466 | 1 694     | 0.8868 |
| 15            | XGBoost  | 0.5247 | 1 689     | 0.8826 |
| 20            | XGBoost  | 0.6831 | 1 670     | 0.8819 |
| **30**        | **XGBoost** | **0.8613** | **1 661** | **0.8856** |

### Figuras generadas automáticamente

| Coste Financiero                     | AUPRC                          |
|--------------------------------------|--------------------------------|
| ![Coste vs factor](results/Figure%202025-11-04%20211504.png) | ![AUPRC vs factor](results/Figure%202025-11-04%20211645.png) |

> **Ganador:** `XGBoost` + `amount_factor=30`  
> **Coste mínimo:** **€1 661**  
> **Ahorro:** 17 % vs RandomForest (€1 997)  
> **Fraude recuperado:** **97.24 %** de $60 127.97

## Paquetes Python

```bash
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```
