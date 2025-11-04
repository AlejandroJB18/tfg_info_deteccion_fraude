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