"""
Utilities Module: Model Persistence, Full Training Orchestration,
and Batch Prediction Pipeline.
"""

import os
import joblib
import pandas as pd
import numpy as np

from src.cost_matrix import CostMatrix
from src.preprocessor import DataPreprocessor
from src.models import FraudDetectionModels, CostSensitiveEnsemble
from src.evaluator import ModelEvaluator


def get_default_model_dir():
    """Returns absolute path to models/ directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_model_artifacts(artifacts, filename="fraud_detection_bundle.joblib"):
    """Saves complete model bundle to disk."""
    save_path = os.path.join(get_default_model_dir(), filename)
    joblib.dump(artifacts, save_path)
    return save_path


def load_model_artifacts(filename="fraud_detection_bundle.joblib"):
    """Loads saved model bundle from disk if available."""
    save_path = os.path.join(get_default_model_dir(), filename)
    if os.path.exists(save_path):
        return joblib.load(save_path)
    return None


def run_full_training_pipeline(df, cost_fp=15.0, cost_fn=250.0, use_undersampling=True, legit_to_fraud_ratio=15):
    """
    Executes end-to-end training:
    1. Preprocessing and feature scaling
    2. Imbalance management (controlled undersampling / weighting)
    3. Model construction (LR, RF, XGB, Cost-Sensitive Ensemble)
    4. Model evaluation and financial cost optimization
    5. Artifact bundle packaging
    """
    # 1. Initialize Cost Matrix & Preprocessor
    cost_matrix = CostMatrix(cost_fp=cost_fp, cost_fn=cost_fn)
    preprocessor = DataPreprocessor(test_size=0.2, val_size=0.15, random_state=42)
    split_data = preprocessor.prepare_data(df)

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    # 2. Imbalance handling for training data
    if use_undersampling:
        X_train_fit, y_train_fit = preprocessor.apply_undersampling(
            X_train, y_train, legit_to_fraud_ratio=legit_to_fraud_ratio
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    # 3. Build Base Models
    factory = FraudDetectionModels(cost_fp=cost_fp, cost_fn=cost_fn, random_state=42)
    lr_standard = factory.build_logistic_regression(cost_sensitive=False)
    lr_cost = factory.build_logistic_regression(cost_sensitive=True)
    rf_cost = factory.build_random_forest(cost_sensitive=True, n_estimators=60, max_depth=10)
    xgb_cost = factory.build_xgboost(cost_sensitive=True, n_estimators=80, max_depth=5)

    # Fit base models
    lr_standard.fit(X_train_fit, y_train_fit)
    lr_cost.fit(X_train_fit, y_train_fit)
    rf_cost.fit(X_train_fit, y_train_fit)
    xgb_cost.fit(X_train_fit, y_train_fit)

    # 4. Build and Fit Cost-Sensitive Ensemble
    ensemble = CostSensitiveEnsemble(cost_matrix=cost_matrix, random_state=42)
    ensemble.fit(X_train_fit, y_train_fit, X_val=X_val, y_val=y_val)

    # Models dictionary for evaluation
    models_dict = {
        "Baseline Logistic Regression": lr_standard,
        "Cost-Sensitive Logistic Regression": lr_cost,
        "Cost-Sensitive Random Forest": rf_cost,
        "Cost-Sensitive XGBoost": xgb_cost,
        "Cost-Sensitive Ensemble": ensemble
    }

    # Set individual cost-optimal thresholds for base models based on validation data
    for name, model in models_dict.items():
        if name != "Baseline Logistic Regression" and hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
            opt_res = cost_matrix.find_optimal_threshold(y_val, val_probs)
            model.optimal_threshold = opt_res["optimal_threshold"]
        elif name == "Baseline Logistic Regression":
            model.optimal_threshold = 0.5

    # 5. Evaluate All Models
    evaluator = ModelEvaluator(cost_matrix=cost_matrix)
    evaluations, comparison_df = evaluator.compare_models(
        models_dict, X_test, y_test, amounts=split_data.get("original_amounts_test")
    )

    artifacts = {
        "models": models_dict,
        "best_model": ensemble,
        "preprocessor": preprocessor,
        "cost_matrix": cost_matrix,
        "evaluations": evaluations,
        "comparison_df": comparison_df,
        "split_data": split_data
    }

    save_model_artifacts(artifacts)
    return artifacts


def predict_batch_transactions(df_input, ensemble_model, preprocessor, threshold=None):
    """
    Runs batch inference on a DataFrame of transactions, adding probability,
    predicted label, risk category, and financial audit flags.
    """
    df_result = df_input.copy()
    feature_cols = [c for c in df_input.columns if c != "Class"]

    # Preprocess
    df_temp = df_input[feature_cols].copy()
    if "Amount" in df_temp.columns:
        df_temp["scaled_amount"] = preprocessor.scaler_amount.transform(df_temp[["Amount"]])
        df_temp = df_temp.drop(columns=["Amount"])
    if "Time" in df_temp.columns:
        df_temp["scaled_time"] = preprocessor.scaler_time.transform(df_temp[["Time"]])
        df_temp = df_temp.drop(columns=["Time"])

    # Ensure feature alignment
    df_temp = df_temp[preprocessor.feature_names]
    X_input = df_temp.values

    # Probabilities & Predictions
    probs = ensemble_model.predict_proba(X_input)[:, 1]
    th = threshold if threshold is not None else ensemble_model.optimal_threshold
    preds = (probs >= th).astype(int)

    df_result["Fraud_Probability"] = np.round(probs, 4)
    df_result["Predicted_Label"] = preds
    df_result["Prediction"] = np.where(preds == 1, "FRAUD", "LEGITIMATE")

    # Risk Categories
    risk_labels = []
    for p in probs:
        if p >= 0.70:
            risk_labels.append("CRITICAL")
        elif p >= th:
            risk_labels.append("HIGH")
        elif p >= 0.10:
            risk_labels.append("MEDIUM")
        else:
            risk_labels.append("LOW")
    df_result["Risk_Level"] = risk_labels

    return df_result


def get_preset_demo_transactions():
    """
    Returns realistic predefined transactions for interactive demonstration.
    """
    # Typical legitimate transaction
    legit_txn = {
        "Time": 45000.0,
        "Amount": 42.50,
        "V1": -0.23, "V2": 0.12, "V3": 1.45, "V4": -0.15,
        "V5": 0.32, "V6": 0.05, "V7": 0.28, "V8": 0.04,
        "V9": 0.18, "V10": -0.05, "V11": 0.42, "V12": -0.12,
        "V13": -0.30, "V14": 0.15, "V15": 0.88, "V16": 0.12,
        "V17": -0.08, "V18": 0.05, "V19": -0.12, "V20": 0.02,
        "V21": -0.05, "V22": 0.15, "V23": 0.02, "V24": 0.10,
        "V25": -0.20, "V26": 0.11, "V27": 0.01, "V28": 0.02
    }

    # High-risk fraudulent transaction with characteristic negative shifts on V10, V12, V14, V17
    fraud_txn = {
        "Time": 82000.0,
        "Amount": 845.00,
        "V1": -3.04, "V2": 2.85, "V3": -4.22, "V4": 3.89,
        "V5": -2.78, "V6": -1.45, "V7": -3.85, "V8": 1.95,
        "V9": -2.65, "V10": -5.82, "V11": 4.12, "V12": -6.95,
        "V13": 0.42, "V14": -8.15, "V15": -0.32, "V16": -4.25,
        "V17": -7.42, "V18": -2.15, "V19": 1.85, "V20": 0.65,
        "V21": 0.85, "V22": -0.35, "V23": -0.25, "V24": 0.12,
        "V25": 0.45, "V26": 0.35, "V27": 0.85, "V28": 0.38
    }

    return {"Legitimate Example (Grocery Store $42.50)": legit_txn, "Fraudulent Example (Compromised Card $845.00)": fraud_txn}
