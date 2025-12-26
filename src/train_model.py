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
    weights[fraud] = amount_train[fraud] * amount_factor

    # 2. Configuración de parámetros por defecto para cada modelo
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

    # 3. Selección e instanciación del modelo fusionando defaults con kwargs
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

    # 4. Lógica común: Entrenamiento
    model.fit(X_train, y_train, sample_weight=weights)
    print(f"Model {model_type.upper()} trained (factor={amount_factor})")
    
    return model

def optimize_params(X_train, y_train, model_type='xgb', n_iter=10, cv=3):
    """
    Realiza una búsqueda aleatoria (RandomizedSearchCV) para encontrar 
    los mejores hiperparámetros estructurales del modelo.
    
    Nota: Optimizamos para AUPRC (Average Precision) que es robusto para desbalanceo.
    """
    print(f"--- Optimizando hiperparámetros para {model_type} ---")
    
    # Espacios de búsqueda
    param_grids = {
        'rf': {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [None, 2, 5, 10, 15, 20, 50],
            'min_samples_leaf': [11, 2, 3, 4, 5, 10, 20],
            'class_weight': ['balanced', 'balanced_subsample']
        },
        'xgb': {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [2, 3, 5, 7, 10, 15, 20],
            'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
        },
        'lgbm': {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [-1, 2, 5, 7, 10, 20],
            'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            'num_leaves': [10, 20, 30, 50, 75, 100],
        }
    }
    
    if model_type not in param_grids:
        print("Modelo no configurado para optimización.")
        return {}

    # Instanciamos el modelo base
    if model_type == 'rf':
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_type == 'xgb':
        clf = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    elif model_type == 'lgbm':
        clf = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

    # Configurar RandomizedSearchCV
    # Usamos AUPRC (average_precision) porque es mejor para fraude que F1 o Accuracy
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grids[model_type],
        n_iter=n_iter,
        scoring='average_precision', 
        cv=StratifiedKFold(n_splits=cv),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print(f"Mejores params ({model_type}): {search.best_params_}")
    print(f"Mejor Score (AUPRC): {search.best_score_:.4f}")
    
    return search.best_params_