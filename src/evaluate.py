# src/evaluate.py
"""
Financial cost and best threshold utilities.
"""

import numpy as np
from sklearn.metrics import average_precision_score

def expected_cost(y_true, y_proba, amounts, threshold=0.5, fraud_loss=0.9, inspect_cost=5):
    """
    Expected financial cost for a given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    cost = (amounts[fn] * fraud_loss).sum() + fp.sum() * inspect_cost
    return cost

def best_threshold_by_cost(y_true, y_proba, amounts, steps=100, fraud_loss=0.9, inspect_cost=5):
    """
    Scan 100 thresholds and return the one with lowest cost.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    costs = [expected_cost(y_true, y_proba, amounts, t, fraud_loss, inspect_cost) for t in thresholds]    
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]

def print_metrics(y_true, y_proba, amounts, model_name):
    """
    Print cost + AUPRC for a model.
    """
    thr, cost = best_threshold_by_cost(y_true, y_proba, amounts)
    auprc = average_precision_score(y_true, y_proba)
    print(f"{model_name}")
    print(f"   Best threshold : {thr:.4f}")
    print(f"   Expected cost  : €{cost:,.0f}")
    print(f"   AUPRC          : {auprc:.4f}")

# ==============================================================================
# PARTE 2: CREDIT SCORING - EVALUACIÓN DE MODELOS (XGB, LGBM, RF)
# ==============================================================================

# --- 1. FUNCIÓN DE COSTE BANCARIO ---
def credit_scoring_cost(y_true, y_proba, amounts, threshold =0.5, LGD=0.8, interest_rate=0.15):
    """
    Calcula el coste financiero para un banco.
    
    Coste Total = (Pérdida por Impago [FN]) + (Coste de Oportunidad [FP])
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # FN (Falso Negativo): El modelo aprueba (0) y el cliente no paga (1)
    # Coste: Perdemos el capital prestado ajustado por la LGD (Loss Given Default)
    fn_mask = (y_true == 1) & (y_pred == 0)
    capital_loss = (amounts[fn_mask] * LGD).sum()
    
    # FP (Falso Positivo): El modelo rechaza (1) y el cliente hubiera pagado (0)
    # Coste: Dejamos de ganar los intereses del préstamo
    fp_mask = (y_true == 0) & (y_pred == 1)
    opportunity_cost = (amounts[fp_mask] * interest_rate).sum()
    
    return capital_loss + opportunity_cost

def best_threshold_credit(y_true, y_proba, amounts, LGD=0.8, interest_rate=0.15):
    """Encuentra el umbral que minimiza el coste financiero bancario."""
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = [
        credit_scoring_cost(y_true, y_proba, amounts, t, LGD, interest_rate) 
        for t in thresholds
    ]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]