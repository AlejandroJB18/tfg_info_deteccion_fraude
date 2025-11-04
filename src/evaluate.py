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

def best_threshold_by_cost(y_true, y_proba, amounts, steps=100):
    """
    Scan 100 thresholds and return the one with lowest cost.
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    costs = [expected_cost(y_true, y_proba, amounts, t) for t in thresholds]
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
