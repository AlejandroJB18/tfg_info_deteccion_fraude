# tfg_info_deteccion_fraude
Este proyecto de Trabajo de Fin de Grado (TFG) desarrolla un sistema de Machine Learning que no solo clasifica transacciones como fraudulentas o legítimas, sino que **minimiza la pérdida financiera total esperada** para una entidad bancaria. El modelo debe aprender que un fraude de 10,000€ es significativamente más costoso que 10 fraudes de 100€.

### Enfoque Técnico
* **Problema Imbalanceado:** Tratamiento de la baja tasa de fraude (clase minoritaria).
* **Sensibilidad al Coste:** Incorporación del campo `Amount` (monto) como factor de penalización durante el entrenamiento y la evaluación.
* **Métrica Clave:** **Costo Financiero Esperado** (Expected Financial Cost), en lugar de F1-Score o Balance Accuracy.

---

## 📂 Estructura y Código

El código se diseña para una ejecución  **secuencial**, adecuada para entornos de desarrollo y depuración como **Visual Studio** o **Spyder**. El código se ha organizado para separar la lógica reutilizable (`src/`) de los experimentos y análisis finales (`notebooks/`).

```plaintext
/data/                                          → Datasets (credit_card.csv, cs_training, etc.)
/exploracion/                                   → Notebooks de la fase inicial de investigación
    ├── modelos_convencionales_german...ipynb      
    ├── modelos_convencionales_varios....ipynb     
    └── modelos_avanzados.ipynb      
/notebooks/                                     → Análisis finales
    ├── analisis_financiero.ipynb               # Simulación y optimización del modelo
    ├── analisis_sensibilidad.ipynb             # Análisis de robustez de negocio
    └── analisis_xai.ipynb                      # Análisis de la explicabilidad del modelo
/models/                                        → Modelos entrenados
/results/                                       → Modelos, métricas y gráficos
/src/                                           → Módulos Python
    ├── load_data.py                            # Carga y limpieza de datos
    ├── train_model.py                          # Entrenamiento con coste variable
    ├── evaluate.py                             # Función de Coste Financiero y Optimización de Umbral
    ├── compare_models.py                       # Simulaciones
    └── benchmark_utils.py                      # Funciones de entrenamiento de benchmarks para modelos
README.md                                       → Este documento
```
### Ejemplos de código

```python
# 1. Cargar datos y separar importes
from src.load_data import load_fraud_csv
df, X, y = load_fraud_csv('data/credit_card.csv')

# 2. Entrenar modelo con penalización variable (Amount * factor)
from src.train_model import train_xgb_with_cost
# amount_factor=20 penaliza 20 veces más los errores en fraudes de alto valor
xgb = train_xgb_with_cost(X_train, y_train, amount_train, amount_factor=20)

# 3. Encontrar el umbral que minimiza el coste real
from src.evaluate import best_threshold_by_cost
# Coste = (FN * 90% del Importe) + (FP * 5€ inspección)
best_thr, min_cost = best_threshold_by_cost(y_test, proba, amount_test)
```
Dataset Credit Card Fraud Detection de Kaggle [data/credit_card.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
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
| 2             | XGBoost  | 0.8217 | 1 704     | 0.8861 |
| 5             | XGBoost  | 0.4951 | 1 683     | 0.8891 |
| 10            | XGBoost  | 0.1783 | 1 737     | 0.8858 |
| **20**        | XGBoost  | 0.7821 | 1 660     | 0.8899 |
| 30            | XGBoost  | 0.8217 | 1 660     | 0.8910 |

El **análisis de sensibilidad** (notebooks/analisis_sensibilidad.ipynb) demostró que el umbral por defecto (0.5) es incorrecto financieramente. El umbral óptimo depende del coste de inspección manual (FP):
- Si inspeccionar cuesta < 3€: El modelo debe ser agresivo (Umbral bajo ~0.40).
- Si inspeccionar cuesta ≥ 3€: El modelo debe ser conservador (Umbral alto 0.7821).
Dado un coste de inspección realista de 5€, la estrategia óptima es conservadora (Umbral 0.78), minimizando las falsas alarmas.

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
- Ejecuta notebooks/analisis_financiero.ipynb para descargar los datos y ver la optimización.
- Ejecuta notebooks/analisis_sensibilidad.ipynb para ver los mapas de calor de decisión y generar el modelo final en models/.

