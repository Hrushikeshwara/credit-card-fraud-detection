"""
Data Loading and Inspection Module for Credit Card Fraud Detection.
Handles Kaggle Credit Card Fraud dataset loading, sampling, and synthetic fallback.
"""

import os
import pandas as pd
import numpy as np


def get_default_data_paths():
    """Returns dict of standard data file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    return {
        "data_dir": data_dir,
        "sample": os.path.join(data_dir, "creditcard_sample.csv"),
        "full": os.path.join(data_dir, "creditcard.csv"),
        "test_batch": os.path.join(data_dir, "test_batch_sample.csv")
    }


def generate_synthetic_fraud_data(n_samples=5000, fraud_ratio=0.005, random_state=42):
    """
    Generates realistic synthetic credit card transactions matching Kaggle distributions
    as a robust fallback if local files cannot be loaded.
    """
    np.random.seed(random_state)
    n_fraud = max(5, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_time = np.sort(np.random.uniform(0, 172800, n_legit))
    legit_features = np.random.normal(0, 1, (n_legit, 28))
    legit_amount = np.random.exponential(scale=70, size=n_legit) + 1.0

    # Fraudulent transactions: feature shifts on V10, V12, V14, V17
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    fraud_features = np.random.normal(0, 1.5, (n_fraud, 28))
    fraud_features[:, 9] -= 4.0   # V10
    fraud_features[:, 11] -= 5.0  # V12
    fraud_features[:, 13] -= 6.0  # V14
    fraud_features[:, 16] -= 4.5  # V17
    fraud_amount = np.random.choice(
        [np.random.uniform(1, 20), np.random.uniform(100, 1500)],
        size=n_fraud,
        p=[0.4, 0.6]
    )

    feature_cols = [f"V{i}" for i in range(1, 29)]
    all_cols = ["Time"] + feature_cols + ["Amount", "Class"]

    legit_data = np.column_stack([legit_time, legit_features, legit_amount, np.zeros(n_legit)])
    fraud_data = np.column_stack([fraud_time, fraud_features, fraud_amount, np.ones(n_fraud)])

    all_data = np.vstack([legit_data, fraud_data])
    np.random.shuffle(all_data)

    df = pd.DataFrame(all_data, columns=all_cols)
    df["Class"] = df["Class"].astype(int)
    return df


def load_credit_card_data(filepath=None, use_sample=True, max_rows=None):
    """
    Loads the credit card fraud dataset.
    Prioritizes:
      1. Provided filepath
      2. creditcard_sample.csv (if use_sample=True)
      3. creditcard.csv (full dataset)
      4. Synthetic generator (fallback)
    """
    paths = get_default_data_paths()

    # Determine which path to load
    target_path = filepath
    if target_path is None:
        if use_sample and os.path.exists(paths["sample"]):
            target_path = paths["sample"]
        elif os.path.exists(paths["full"]):
            target_path = paths["full"]
        elif os.path.exists(paths["sample"]):
            target_path = paths["sample"]

    if target_path and os.path.exists(target_path):
        if max_rows:
            df = pd.read_csv(target_path, nrows=max_rows)
        else:
            df = pd.read_csv(target_path)
    else:
        # Fallback to realistic synthetic generation
        df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.01)

    # Clean target column type
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)

    return df


def get_dataset_statistics(df):
    """
    Computes key summary statistics for the dataset,
    including class imbalance, amounts, and missing values.
    """
    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    total_records = len(df)
    class_counts = df[target_col].value_counts().to_dict()
    fraud_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    fraud_ratio = (fraud_count / total_records * 100) if total_records > 0 else 0

    stats = {
        "total_records": total_records,
        "total_features": len(df.columns) - 1,
        "fraud_count": fraud_count,
        "legitimate_count": legit_count,
        "fraud_percentage": fraud_ratio,
        "imbalance_ratio": f"1:{int(round(legit_count / fraud_count))}" if fraud_count > 0 else "N/A",
        "missing_values": int(df.isnull().sum().sum()),
        "amount_stats": {
            "legit_mean": float(df[df[target_col] == 0]["Amount"].mean()) if legit_count > 0 else 0.0,
            "legit_median": float(df[df[target_col] == 0]["Amount"].median()) if legit_count > 0 else 0.0,
            "fraud_mean": float(df[df[target_col] == 1]["Amount"].mean()) if fraud_count > 0 else 0.0,
            "fraud_median": float(df[df[target_col] == 1]["Amount"].median()) if fraud_count > 0 else 0.0,
            "fraud_total_loss": float(df[df[target_col] == 1]["Amount"].sum()) if fraud_count > 0 else 0.0
        }
    }
    return stats
