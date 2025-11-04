# tfg_info_deteccion_fraude
Desarrollar un sistema de Machine Learning que no solo clasifique transacciones como fraudulentas o legítimas, sino que **minimice la pérdida financiera total esperada**. El modelo debe aprender que un fraude de $10,000 es significativamente más costoso que 10 fraudes de $100.

### Enfoque Técnico
* **Problema Imbalanceado:** Tratamiento de la baja tasa de fraude (clase minoritaria).
* **Sensibilidad al Coste:** Incorporación del campo `Amount` (monto) como factor de penalización durante el entrenamiento y la evaluación.
* **Métrica Clave:** **Costo Financiero Esperado** (Expected Financial Cost), en lugar de F1-Score o Accuracy.

---

## 📂 Estructura y Código

El código se diseña para una ejecución  **secuencial**, adecuada para entornos de desarrollo y depuración como **Visual Studio** o **Spyder**.

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
| ![Coste vs factor](results/Figure_2025-11-04_211504.png) | ![AUPRC vs factor](results/Figure_2025-11-04_211645.png) |

> **Ganador:** `XGBoost` + `amount_factor=30`  
> **Coste mínimo:** **€1 661**  
> **Ahorro:** 17 % vs RandomForest (€1 997)  
> **Fraude recuperado:** **97.24 %** de $60 127.97

## Paquetes Python

```bash
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
```
