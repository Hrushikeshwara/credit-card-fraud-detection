"""
Preprocessing and Class Imbalance Handling Module.
Performs feature scaling (RobustScaler), train-test splitting,
and class balancing strategies.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class DataPreprocessor:
    """
    Handles feature scaling, train-test splitting, and imbalance treatment
    for Credit Card Fraud Detection.
    """

    def __init__(self, test_size=0.2, val_size=0.1, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler_amount = RobustScaler()
        self.scaler_time = RobustScaler()
        self.feature_names = None
        self.is_fitted = False

    def prepare_data(self, df, target_col="Class"):
        """
        Extracts features (X) and target (y), scales Time and Amount,
        and splits into train, validation, and test partitions.
        """
        df_clean = df.copy()

        # Separate target, features, and raw amounts
        y = df_clean[target_col].astype(int).values
        raw_amounts = df_clean["Amount"].values if "Amount" in df_clean.columns else np.zeros(len(df_clean))
        X_df = df_clean.drop(columns=[target_col])

        # Scale Amount and Time if present
        if "Amount" in X_df.columns:
            X_df["scaled_amount"] = self.scaler_amount.fit_transform(X_df[["Amount"]])
            X_df = X_df.drop(columns=["Amount"])
        if "Time" in X_df.columns:
            X_df["scaled_time"] = self.scaler_time.fit_transform(X_df[["Time"]])
            X_df = X_df.drop(columns=["Time"])

        self.feature_names = list(X_df.columns)
        self.is_fitted = True
        X = X_df.values

        # Stratified train/test split including raw amounts
        X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
            X, y, raw_amounts, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Further split train into train and validation
        if self.val_size > 0:
            val_ratio = self.val_size / (1.0 - self.test_size)
            X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
                X_train, y_train, amt_train, test_size=val_ratio, random_state=self.random_state, stratify=y_train
            )
        else:
            X_val, y_val, amt_val = None, None, None

        return {
            "X_train": X_train,
            "y_train": y_train,
            "amt_train": amt_train,
            "X_val": X_val,
            "y_val": y_val,
            "amt_val": amt_val,
            "X_test": X_test,
            "y_test": y_test,
            "amt_test": amt_test,
            "feature_names": list(X_df.columns),
            "original_amounts_test": amt_test
        }

    def transform_single_transaction(self, raw_dict):
        """
        Transforms a raw dictionary of transaction features into the model input vector.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")

        df_single = pd.DataFrame([raw_dict])
        if "Amount" in df_single.columns:
            df_single["scaled_amount"] = self.scaler_amount.transform(df_single[["Amount"]])
            df_single = df_single.drop(columns=["Amount"])
        if "Time" in df_single.columns:
            df_single["scaled_time"] = self.scaler_time.transform(df_single[["Time"]])
            df_single = df_single.drop(columns=["Time"])

        # Retain original feature ordering
        df_single = df_single[self.feature_names]
        return df_single.values

    def apply_undersampling(self, X_train, y_train, legit_to_fraud_ratio=10):
        """
        Balances training data by controlled under-sampling of majority class
        to achieve desired legit:fraud ratio (default 10:1).
        """
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]

        n_fraud = len(fraud_idx)
        n_legit_keep = min(len(legit_idx), n_fraud * legit_to_fraud_ratio)

        np.random.seed(self.random_state)
        selected_legit_idx = np.random.choice(legit_idx, size=n_legit_keep, replace=False)

        balanced_idx = np.concatenate([fraud_idx, selected_legit_idx])
        np.random.shuffle(balanced_idx)

        return X_train[balanced_idx], y_train[balanced_idx]
