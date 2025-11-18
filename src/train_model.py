# src/train_model.py
"""
Cost-sensitive trainers: RandomForest & XGBoost.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_rf_with_cost(X_train, y_train, amount_train, amount_factor=10):
    """
    Train RandomForest with fraud amount as sample weight.
    """
    weights = np.ones(len(y_train))
    fraud = y_train == 1
    weights[fraud] = amount_train[fraud] * amount_factor

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train, sample_weight=weights)
    print(f"RandomForest trained (factor={amount_factor})")
    return rf

def train_lgbm_with_cost(X_train, y_train, amount_train, amount_factor=10):
    """
    Train LightGBM with fraud amount as sample weight.
    """
    weights = np.ones(len(y_train))
    fraud = y_train == 1
    weights[fraud] = amount_train[fraud] * amount_factor

    # 2. Configurar el modelo LightGBM
    lgbm = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        objective='binary',          # Especificamos que es clasificación binaria
        metric='binary_logloss',     # Equivalente a 'logloss' de XGBoost
        verbose=-1                   # Reduce los warnings de consola
    )


    lgbm.fit(X_train, y_train, sample_weight=weights)
    print(f"LightGBM trained (factor={amount_factor})")
    return lgbm

def train_xgb_with_cost(X_train, y_train, amount_train, amount_factor=10):
    """
    Train XGBoost with fraud amount as sample weight.
    """
    weights = np.ones(len(y_train))
    fraud = y_train == 1
    weights[fraud] = amount_train[fraud] * amount_factor

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train, sample_weight=weights)
    print(f"XGBoost trained (factor={amount_factor})")
    return xgb