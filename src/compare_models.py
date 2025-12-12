# src/compare_models.py
"""
Full experiment: train RF & XGB for several amount_factors.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.load_data import load_fraud_csv
from src.train_model import train_model_with_cost
from src.evaluate import print_metrics, best_threshold_by_cost

def compare_all_factors(csv_path='data/creditcard.csv'):
    """
    Run the complete comparison and plot results.
    """
    df, X, y = load_fraud_csv(csv_path)
    amount = X['Amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    amount_train = X_train['Amount']
    amount_test  = X_test['Amount']

    factors = [5, 10, 15, 20, 30]
    rf_costs = []
    xgb_costs = []
    lgbm_costs = []

    print("\n" + "="*60)
    print("COST-SENSITIVE COMPARISON")
    print("="*60)

    for f in factors:
        print(f"\n--- amount_factor = {f} ---")
        rf = train_model_with_cost(X_train, y_train, amount_train, model_type="rf", amount_factor = f)
        xgb = train_model_with_cost(X_train, y_train, amount_train, model_type="xgb", amount_factor = f)
        lgbm = train_model_with_cost(X_train, y_train, amount_train, model_type="lgbm", amount_factor = f)

        rf_proba = rf.predict_proba(X_test)[:, 1]
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        lgbm_proba = lgbm.predict_proba(X_test)[:, 1]

        rf_thr, rf_cost = best_threshold_by_cost(y_test, rf_proba, amount_test)
        xgb_thr, xgb_cost = best_threshold_by_cost(y_test, xgb_proba, amount_test)

        rf_costs.append(rf_cost)
        xgb_costs.append(xgb_cost)
        lgbm_costs.append(lgbm_cost)

        print_metrics(y_test, rf_proba, amount_test, "RandomForest")
        print_metrics(y_test, xgb_proba, amount_test, "XGBoost")
        print_metrics(y_test, lgbm_proba, amount_test, "LightGBM")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(factors, rf_costs, 'o-', label='RandomForest')
    plt.plot(factors, xgb_costs, 's-', label='XGBoost')
    plt.title('Expected Financial Cost vs amount_factor')
    plt.xlabel('amount_factor')
    plt.ylabel('Cost (€)')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_f = factors[np.argmin(xgb_costs)]
    print(f"\nBEST CONFIG: XGBoost + amount_factor={best_f}")
    print(f"   Minimum cost: €{min(xgb_costs):,.0f}")

# Ejecuta esto en Spyder para lanzar todo:
# from src.compare_models import compare_all_factors
# compare_all_factors('data/creditcard.csv')