# src/train_model.py
"""
Cost-sensitive trainers: RandomForest & XGBoost.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_model_with_cost(X_train, y_train, amount_train, model_type='xgb', amount_factor=10, **kwargs):
    """
    Train a classifier (RF, XGB, LGBM) with fraud amount as sample weight.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        amount_train: Vector con los montos de las transacciones.
        model_type (str): Tipo de modelo ('rf', 'xgb', 'lgbm').
        amount_factor (int): Factor multiplicativo para el peso del fraude.
        **kwargs: Parámetros adicionales para el modelo (sobreescriben los defaults).
    """
    
    # 1. Lógica común: Calcular pesos basados en el monto
    weights = np.ones(len(y_train))
    fraud = y_train == 1
    weights[fraud] = amount_train[fraud] * amount_factor

    # 2. Configuración de parámetros por defecto para cada modelo
    # Estos son los valores que tenías en tus funciones originales
    rf_params = {
        'n_estimators': 200,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }

    lgbm_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1
    }

    # 3. Selección e instanciación del modelo
    if model_type == 'rf':
        rf_params.update(kwargs) # Actualiza con parámetros extra si existen
        model = RandomForestClassifier(**rf_params)
        
    elif model_type == 'xgb':
        xgb_params.update(kwargs)
        model = XGBClassifier(**xgb_params)
        
    elif model_type == 'lgbm':
        lgbm_params.update(kwargs)
        model = LGBMClassifier(**lgbm_params)
        
    else:
        raise ValueError(f"Modelo '{model_type}' no soportado. Usa 'rf', 'xgb' o 'lgbm'.")

    # 4. Lógica común: Entrenamiento
    model.fit(X_train, y_train, sample_weight=weights)
    print(f"Model {model_type.upper()} trained (factor={amount_factor})")
    
    return model