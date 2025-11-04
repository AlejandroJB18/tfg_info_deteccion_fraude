# tfg_info_deteccion_fraude
Desarrollar un sistema de Machine Learning que no solo clasifique transacciones como fraudulentas o legítimas, sino que **minimice la pérdida financiera total esperada**. El modelo debe aprender que un fraude de $10,000 es significativamente más costoso que 10 fraudes de $100.

### Enfoque Técnico
* **Problema Imbalanceado:** Tratamiento de la baja tasa de fraude (clase minoritaria).
* **Sensibilidad al Coste:** Incorporación del campo `Amount` (monto) como factor de penalización durante el entrenamiento y la evaluación.
* **Métrica Clave:** **Costo Financiero Esperado** (Expected Financial Cost), en lugar de F1-Score o Accuracy.

---

## 📂 Estructura y Código

El flujo de trabajo es **secuencial** y está diseñado para una ejecución limpia y legible, adecuada para entornos de desarrollo y depuración como **Spyder**.

```plaintext
/data/          → Datasets (creditcard.csv, etc.)
/src/           → Módulos Python
    ├── load_data.py
    ├── train_model.py
    ├── evaluate.py
    └── compare_models.py
/notebooks/     → Análisis exploratorio (opcional)
/results/       → Modelos, métricas y gráficos
```
### Ejemplos de código

```python
# Cargar datos
from src.load_data import load_fraud_csv
df, X, y = load_fraud_csv('data/creditcard.csv')
amount = X['Amount']
```
```python
# Entrenar modelo sensible al coste
from src.train_model import train_rf_with_cost
rf = train_rf_with_cost(X_train, y_train, amount_train, amount_factor=15)
```
```python
# Evaluar con coste real
from src.evaluate import expected_cost, best_threshold_by_cost
cost = expected_cost(y_test, proba, amount_test)
print(f'Costo esperado: €{cost:,.2f}')
```

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

## Posibles Experimentos a Realizar

1. Variación de `amount_factor` (5, 10, 15, 20)  
2. Reequilibrado por SMOTE + peso por monto  
3. Undersampling ponderado  
4. Redes neuronales con *custom loss*

> **Ejemplo de conclusión:** `XGBoost` + `amount_factor=15` ofrece el mejor balance coste/rendimiento.
## Paquetes Python

```bash
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```
