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


def load_german_data(filepath='../data/german_credit_data.csv'):
    """
    Carga y preprocesa el dataset German Credit para análisis de costes.
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # Cargar dataset (asegúrate de tener el CSV o usar la librería directa)
    # Si no tienes el csv descargado, usa sklearn o descárgalo de UCI
    try:
        df = pd.read_csv(filepath)
    except:
        # Fallback: Cargar desde URL si no está local
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/german_credit.csv"
        df = pd.read_csv(url)

    # 1. Definir Target
    # En este dataset: 2 = Bad (Impago), 1 = Good (Pagó)
    # Lo convertimos a: 1 = Impago (Clase Positiva/Fraude), 0 = Pagó
    # Ajusta según tu CSV, a veces viene como "bad"/"good" texto
    if 'Cost Matrix(Risk)' in df.columns: # Versión raw a veces varía
        pass 
    
    # Estandarización típica de este dataset
    # Asumimos que la columna target es 'Risk' o similar. 
    # Si usas la versión de Kaggle común:
    if 'Risk' in df.columns:
        df['target'] = df['Risk'].apply(lambda x: 1 if x == 'bad' else 0)
        df = df.drop('Risk', axis=1)
    elif 'class' in df.columns: # Otra versión común
         df['target'] = df['class'].apply(lambda x: 1 if x == 2 else 0)
         df = df.drop('class', axis=1)

    # 2. Gestionar Amount
    # Necesitamos la columna de dinero para tus cálculos financieros
    if 'Credit amount' in df.columns:
        df = df.rename(columns={'Credit amount': 'Amount'})
    elif 'amount' in df.columns:
        df = df.rename(columns={'amount': 'Amount'})
        
    # 3. Codificar variables categóricas (Housing, Purpose, Sex...)
    # XGBoost necesita números, no strings
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(['target'], axis=1)
    y = df['target']
    
    print(f"German Data Cargado: {X.shape}, Tasa de Impago: {y.mean():.2%}")
    return df, X, y