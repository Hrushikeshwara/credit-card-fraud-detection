"""
Model Evaluation and Metrics Module.
Computes:
- Precision, Recall, Specificity, F1-Score
- ROC-AUC, PR-AUC (Average Precision)
- Confusion Matrix (TN, FP, FN, TP)
- Total Misclassification Cost & Financial Savings
- Threshold Curves (ROC, PR, Cost vs Threshold)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    accuracy_score
)


class ModelEvaluator:
    """
    Comprehensive evaluator for imbalanced and cost-sensitive fraud detection models.
    """

    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix

    def evaluate_model(self, model, X_test, y_test, threshold=None, amounts=None, model_name="Model"):
        """
        Evaluates a single model at a specified or optimal decision threshold.
        """
        # Determine prediction probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-y_scores))
        else:
            y_prob = model.predict(X_test).astype(float)

        # Decision threshold
        if threshold is None:
            if hasattr(model, "optimal_threshold"):
                th = model.optimal_threshold
            else:
                th = 0.5
        else:
            th = threshold

        y_pred = (y_prob >= th).astype(int)

        # Metrics computation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            roc_auc = 0.5

        try:
            pr_auc = average_precision_score(y_test, y_prob)
        except Exception:
            pr_auc = 0.0

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Cost calculations
        cost_dict = self.cost_matrix.calculate_total_cost(y_test, y_pred, amounts=amounts)

        # Baseline cost (No model / predict all legit)
        all_legit_pred = np.zeros(len(y_test), dtype=int)
        baseline_cost_dict = self.cost_matrix.calculate_total_cost(y_test, all_legit_pred, amounts=amounts)

        # Cost at standard 0.5 threshold
        standard_pred = (y_prob >= 0.5).astype(int)
        std_cost_dict = self.cost_matrix.calculate_total_cost(y_test, standard_pred, amounts=amounts)

        # ROC and PR curves
        fpr_arr, tpr_arr, roc_th = roc_curve(y_test, y_prob)
        pr_prec_arr, pr_rec_arr, pr_th = precision_recall_curve(y_test, y_prob)

        # Cost vs Threshold curve
        threshold_curve = self.cost_matrix.find_optimal_threshold(y_test, y_prob, amounts=amounts, num_steps=60)

        results = {
            "model_name": model_name,
            "threshold": float(th),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "total_cost": float(cost_dict["total_cost"]),
            "fp_cost": float(cost_dict["fp_cost"]),
            "fn_cost": float(cost_dict["fn_cost"]),
            "standard_threshold_cost": float(std_cost_dict["total_cost"]),
            "baseline_cost": float(baseline_cost_dict["total_cost"]),
            "savings_vs_baseline": float(baseline_cost_dict["total_cost"] - cost_dict["total_cost"]),
            "savings_vs_standard": float(std_cost_dict["total_cost"] - cost_dict["total_cost"]),
            "roc_curve": {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()},
            "pr_curve": {"precision": pr_prec_arr.tolist(), "recall": pr_rec_arr.tolist()},
            "threshold_curve": threshold_curve,
            "y_prob": y_prob
        }
        return results

    def compare_models(self, models_dict, X_test, y_test, amounts=None):
        """
        Runs evaluation across multiple models and outputs a comparative DataFrame.
        """
        evaluations = {}
        comparison_records = []

        for name, model in models_dict.items():
            res = self.evaluate_model(model, X_test, y_test, amounts=amounts, model_name=name)
            evaluations[name] = res

            cm = res["confusion_matrix"]
            comparison_records.append({
                "Model": name,
                "Threshold": f"{res['threshold']:.3f}",
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1-Score": round(res["f1_score"], 4),
                "ROC-AUC": round(res["roc_auc"], 4),
                "PR-AUC": round(res["pr_auc"], 4),
                "TN": cm["TN"],
                "FP": cm["FP"],
                "FN": cm["FN"],
                "TP": cm["TP"],
                "Total Cost ($)": f"${res['total_cost']:,.2f}",
                "Cost Savings vs Baseline ($)": f"${res['savings_vs_baseline']:,.2f}"
            })

        df_comparison = pd.DataFrame(comparison_records)
        return evaluations, df_comparison
