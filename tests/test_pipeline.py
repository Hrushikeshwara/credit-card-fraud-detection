"""
Unit and Integration Tests for Cost-Sensitive Fraud Detection System.
"""

import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data_loader import load_credit_card_data, get_dataset_statistics, generate_synthetic_fraud_data
from src.cost_matrix import CostMatrix
from src.preprocessor import DataPreprocessor
from src.models import FraudDetectionModels, CostSensitiveEnsemble
from src.evaluator import ModelEvaluator
from src.utils import predict_batch_transactions


def test_data_loader_and_synthetic():
    df_synth = generate_synthetic_fraud_data(n_samples=500, fraud_ratio=0.05)
    assert len(df_synth) == 500
    assert "Class" in df_synth.columns
    assert (df_synth["Class"] == 1).sum() > 0

    stats = get_dataset_statistics(df_synth)
    assert stats["total_records"] == 500
    assert stats["fraud_count"] > 0
    assert stats["missing_values"] == 0


def test_cost_matrix_and_bayes_threshold():
    cm = CostMatrix(cost_fp=10.0, cost_fn=190.0)
    # Bayes threshold = 10 / (10 + 190) = 10 / 200 = 0.05
    assert abs(cm.bayes_optimal_threshold - 0.05) < 1e-4

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # TN=1, FP=1, FN=1, TP=1
    res = cm.calculate_total_cost(y_true, y_pred)
    assert res["total_cost"] == 10.0 + 190.0  # FP * 10 + FN * 190 = 200
    assert res["confusion_counts"]["FP"] == 1
    assert res["confusion_counts"]["FN"] == 1


def test_preprocessor_and_split():
    df = generate_synthetic_fraud_data(n_samples=600, fraud_ratio=0.05)
    preprocessor = DataPreprocessor(test_size=0.2, val_size=0.1, random_state=42)
    split_data = preprocessor.prepare_data(df)

    assert "X_train" in split_data
    assert "X_test" in split_data
    assert len(split_data["y_test"]) == 120
    assert preprocessor.is_fitted


def test_models_and_ensemble():
    df = generate_synthetic_fraud_data(n_samples=800, fraud_ratio=0.08)
    cm = CostMatrix(cost_fp=15.0, cost_fn=250.0)
    preprocessor = DataPreprocessor(test_size=0.25, val_size=0.15, random_state=42)
    data = preprocessor.prepare_data(df)

    # Train Cost-Sensitive Ensemble
    ensemble = CostSensitiveEnsemble(cost_matrix=cm, random_state=42)
    ensemble.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

    assert ensemble.is_fitted
    probs = ensemble.predict_proba(data["X_test"])
    assert probs.shape == (len(data["y_test"]), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = ensemble.predict(data["X_test"])
    assert len(preds) == len(data["y_test"])

    # Test Evaluator
    evaluator = ModelEvaluator(cost_matrix=cm)
    eval_res = evaluator.evaluate_model(ensemble, data["X_test"], data["y_test"])
    assert "roc_auc" in eval_res
    assert "pr_auc" in eval_res
    assert "total_cost" in eval_res
    assert eval_res["roc_auc"] > 0.70


def test_batch_prediction():
    df = generate_synthetic_fraud_data(n_samples=200, fraud_ratio=0.05)
    cm = CostMatrix(cost_fp=15.0, cost_fn=250.0)
    preprocessor = DataPreprocessor(test_size=0.2, val_size=0.1, random_state=42)
    data = preprocessor.prepare_data(df)

    ensemble = CostSensitiveEnsemble(cost_matrix=cm, random_state=42)
    ensemble.fit(data["X_train"], data["y_train"])

    results = predict_batch_transactions(df, ensemble, preprocessor)
    assert "Fraud_Probability" in results.columns
    assert "Prediction" in results.columns
    assert "Risk_Level" in results.columns
    assert set(results["Prediction"].unique()).issubset({"LEGITIMATE", "FRAUD"})


if __name__ == "__main__":
    print("Running tests...")
    test_data_loader_and_synthetic()
    print("test_data_loader_and_synthetic passed.")
    test_cost_matrix_and_bayes_threshold()
    print("test_cost_matrix_and_bayes_threshold passed.")
    test_preprocessor_and_split()
    print("test_preprocessor_and_split passed.")
    test_models_and_ensemble()
    print("test_models_and_ensemble passed.")
    test_batch_prediction()
    print("test_batch_prediction passed.")
    print("ALL 5 UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
