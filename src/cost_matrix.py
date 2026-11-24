"""
Cost Matrix and Financial Cost Optimization Module.
Defines misclassification costs, theoretical Bayes thresholds,
and empirical cost-minimizing threshold search.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


class CostMatrix:
    """
    Encapsulates financial costs associated with classification outcomes:
    - C_fp: Cost of a False Positive (investigation friction, customer SMS/call, alert review)
    - C_fn: Cost of a False Negative (unrecovered fraudulent transaction loss)
    - C_tp: Cost of a True Positive (small handling/admin cost, default $0)
    - C_tn: Cost of a True Negative (business-as-usual, default $0)
    """

    def __init__(self, cost_fp=15.0, cost_fn=250.0, cost_tp=0.0, cost_tn=0.0):
        self.cost_fp = float(cost_fp)
        self.cost_fn = float(cost_fn)
        self.cost_tp = float(cost_tp)
        self.cost_tn = float(cost_tn)

    @property
    def bayes_optimal_threshold(self):
        """
        Calculates theoretical Bayes decision threshold:
        P* = (C_fp - C_tn) / [(C_fp - C_tn) + (C_fn - C_tp)]
        """
        numerator = self.cost_fp - self.cost_tn
        denominator = (self.cost_fp - self.cost_tn) + (self.cost_fn - self.cost_tp)
        if denominator <= 0:
            return 0.5
        threshold = numerator / denominator
        return float(np.clip(threshold, 0.001, 0.999))

    def calculate_total_cost(self, y_true, y_pred, amounts=None):
        """
        Computes total misclassification financial cost.
        If amounts array is provided, FN cost can reflect actual transaction amounts.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        if amounts is not None:
            # Transaction-specific false negative cost
            fn_mask = (np.array(y_true) == 1) & (np.array(y_pred) == 0)
            fn_cost_total = float(np.sum(amounts[fn_mask]))
        else:
            fn_cost_total = float(fn * self.cost_fn)

        fp_cost_total = float(fp * self.cost_fp)
        tp_cost_total = float(tp * self.cost_tp)
        tn_cost_total = float(tn * self.cost_tn)

        total_cost = fp_cost_total + fn_cost_total + tp_cost_total + tn_cost_total

        return {
            "total_cost": total_cost,
            "fp_cost": fp_cost_total,
            "fn_cost": fn_cost_total,
            "tp_cost": tp_cost_total,
            "tn_cost": tn_cost_total,
            "confusion_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }

    def find_optimal_threshold(self, y_true, y_prob, amounts=None, num_steps=100):
        """
        Empirically scans probability thresholds in [0.01, 0.99]
        to find the threshold that strictly minimizes total financial loss.
        """
        thresholds = np.linspace(0.01, 0.99, num_steps)
        costs = []
        fp_counts = []
        fn_counts = []

        best_cost = float("inf")
        best_threshold = 0.5

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            cost_dict = self.calculate_total_cost(y_true, y_pred, amounts=amounts)
            cost = cost_dict["total_cost"]
            costs.append(cost)
            fp_counts.append(cost_dict["confusion_counts"]["FP"])
            fn_counts.append(cost_dict["confusion_counts"]["FN"])

            if cost < best_cost:
                best_cost = cost
                best_threshold = float(th)

        # Baseline cost with no model (predict all 0: legitimate)
        all_legit_pred = np.zeros(len(y_true), dtype=int)
        baseline_cost_dict = self.calculate_total_cost(y_true, all_legit_pred, amounts=amounts)
        baseline_cost = baseline_cost_dict["total_cost"]

        # Cost at standard 0.5 threshold
        standard_pred = (y_prob >= 0.5).astype(int)
        standard_cost_dict = self.calculate_total_cost(y_true, standard_pred, amounts=amounts)
        standard_cost = standard_cost_dict["total_cost"]

        savings_vs_baseline = baseline_cost - best_cost
        savings_vs_standard = standard_cost - best_cost

        return {
            "optimal_threshold": best_threshold,
            "minimum_cost": best_cost,
            "standard_cost": standard_cost,
            "baseline_cost": baseline_cost,
            "savings_vs_baseline": savings_vs_baseline,
            "savings_vs_standard": savings_vs_standard,
            "thresholds": thresholds.tolist(),
            "costs": costs,
            "fp_counts": fp_counts,
            "fn_counts": fn_counts,
            "bayes_threshold": self.bayes_optimal_threshold
        }
