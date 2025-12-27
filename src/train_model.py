# src/train_model.py
"""
Cost-sensitive trainers: RandomForest & XGBoost.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, average_precision_score

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

    # 2. CALCULO ROBUSTO DE PESOS
    if np.sum(fraud) > 0:
        fraud_amounts = amount_train[fraud]
        avg_fraud_amount = fraud_amounts.mean()
        
        # Evitamos división por cero por si acaso
        if avg_fraud_amount == 0: 
            avg_fraud_amount = 1.0
            
        # Normalizamos: Cuántas veces es este fraude respecto al promedio
        # Ej: Si promedio es 1000 y fraude es 5000 -> ratio es 5.
        amount_ratio = fraud_amounts / avg_fraud_amount
        
        # Aplicamos el factor sobre el ratio normalizado
        # Si factor es 10, el peso será 5 * 10 = 50. (Manejable para XGBoost)
        weights[fraud] = amount_ratio * amount_factor

    # 3. Configuración de parámetros por defecto para cada modelo
    # Estos son los valores que tenías en tus funciones originales
    rf_defaults = {
        'n_estimators': 200,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    xgb_defaults = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }

    lgbm_defaults = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1
    }

    # 4. Selección e instanciación del modelo fusionando defaults con kwargs
    if model_type == 'rf':
        params = rf_defaults.copy()
        params.update(kwargs) # Sobrescribe defaults con lo que venga de fuera (ej. del GridSearch)
        model = RandomForestClassifier(**params)
        
    elif model_type == 'xgb':
        params = xgb_defaults.copy()
        params.update(kwargs)
        model = XGBClassifier(**params)
        
    elif model_type == 'lgbm':
        params = lgbm_defaults.copy()
        params.update(kwargs)
        model = LGBMClassifier(**params)
        
    else:
        raise ValueError(f"Modelo '{model_type}' no soportado. Usa 'rf', 'xgb' o 'lgbm'.")

    # 5. Lógica común: Entrenamiento
    model.fit(X_train, y_train, sample_weight=weights)
    print(f"Model {model_type.upper()} trained (factor={amount_factor})")
    
    return model

