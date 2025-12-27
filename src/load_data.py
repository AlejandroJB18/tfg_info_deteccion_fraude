#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 21:03:28 2025

@author: fran
"""

# src/load_data.py
"""
Load fraud CSV and return everything needed for cost-sensitive models.
"""

import pandas as pd
from pathlib import Path

def load_fraud_csv(csv_path):
    """
    Load CSV and show basic fraud stats.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    df : full DataFrame
    X  : features (Amount included)
    y  : labels (0=legit, 1=fraud)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df):,} transactions from {csv_path.name}")
    fraud_rate = df['Class'].mean()
    fraud_amount = df[df['Class']==1]['Amount'].sum()
    print(f"Fraud rate: {fraud_rate:.5%}")
    print(f"Total fraud amount: ${fraud_amount:,.2f}")

    X = df.drop('Class', axis=1)
    y = df['Class']
    return df, X, y


def load_credit_scoring_data(filepath='../data/cs-training.csv'):
    """
    Carga el dataset 'Give Me Some Credit', trata nulos y genera una columna 'Amount'
    estimada para el aprendizaje sensible al coste.
    """
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # 1. Cargar datos
    df = pd.read_csv(filepath)
    
    # El dataset suele tener una columna de índice inútil 'Unnamed: 0'
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # 2. Tratamiento de Nulos (Crítico para este dataset)
    # MonthlyIncome y NumberOfDependents suelen tener nulos.
    # XGBoost los maneja, pero para calcular el 'Amount' necesitamos el Income lleno.
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # 3. Generar columna 'Amount' (Proxy de Exposición al Default)
    # Asumimos que la línea de crédito es aprox 12 veces el ingreso mensual
    # Esto es necesario para que funcione tu lógica de 'amount_factor'
    df['Amount'] = df['MonthlyIncome'] * 6 
    
    # Evitar montos 0 para no romper logaritmos o pesos (poner un mínimo de 1000)
    df['Amount'] = df['Amount'].apply(lambda x: max(x, 1000))

    # 4. Separar X e y
    # Target: SeriousDlqin2yrs (1 = Default, 0 = Paga)
    target_col = 'SeriousDlqin2yrs'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"Credit Scoring Data Loaded: {len(df):,} records")
    print(f"Default Rate: {y.mean():.2%}")
    print(f"Total Estimated Exposure: ${df['Amount'].sum():,.0f}")

    return df, X, y