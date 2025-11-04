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
