"""
Machine Learning Models and Cost-Sensitive Ensemble Module.
Implements:
1. Logistic Regression (Standard & Cost-Sensitive)
2. Random Forest (Standard & Cost-Sensitive)
3. XGBoost (Standard & Cost-Sensitive)
4. Cost-Sensitive Ensemble Classifier (Meta-Weight Soft Voting with Cost-Minimizing Thresholding)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class FraudDetectionModels:
    """
    Factory and training manager for baseline and cost-sensitive classifiers.
    """

    def __init__(self, cost_fp=15.0, cost_fn=250.0, random_state=42):
        self.cost_fp = float(cost_fp)
        self.cost_fn = float(cost_fn)
        self.random_state = random_state

        # Compute cost-sensitive weight ratio
        self.cost_ratio = self.cost_fn / max(1e-3, self.cost_fp)
        self.custom_class_weights = {0: 1.0, 1: self.cost_ratio}

    def build_logistic_regression(self, cost_sensitive=True):
        """Constructs Logistic Regression classifier."""
        if cost_sensitive:
            return LogisticRegression(
                class_weight=self.custom_class_weights,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs"
            )
        else:
            return LogisticRegression(
                class_weight=None,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs"
            )

    def build_random_forest(self, cost_sensitive=True, n_estimators=100, max_depth=12):
        """Constructs Random Forest classifier."""
        if cost_sensitive:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=None,
                random_state=self.random_state,
                n_jobs=-1
            )

    def build_xgboost(self, cost_sensitive=True, n_estimators=100, max_depth=6, learning_rate=0.1):
        """Constructs XGBoost classifier."""
        if cost_sensitive:
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=self.cost_ratio,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1
            )
        else:
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=1.0,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1
            )


class CostSensitiveEnsemble:
    """
    Cost-Sensitive Meta-Ensemble combining Logistic Regression, Random Forest, and XGBoost.
    Optimizes base model weights and decision threshold to strictly minimize financial misclassification cost.
    """

    def __init__(self, cost_matrix, random_state=42):
        self.cost_matrix = cost_matrix
        self.random_state = random_state

        factory = FraudDetectionModels(
            cost_fp=cost_matrix.cost_fp,
            cost_fn=cost_matrix.cost_fn,
            random_state=random_state
        )

        self.lr_model = factory.build_logistic_regression(cost_sensitive=True)
        self.rf_model = factory.build_random_forest(cost_sensitive=True, n_estimators=60, max_depth=10)
        self.xgb_model = factory.build_xgboost(cost_sensitive=True, n_estimators=80, max_depth=5)

        self.models = {
            "Logistic Regression": self.lr_model,
            "Random Forest": self.rf_model,
            "XGBoost": self.xgb_model
        }

        self.weights = {"Logistic Regression": 0.20, "Random Forest": 0.40, "XGBoost": 0.40}
        self.optimal_threshold = cost_matrix.bayes_optimal_threshold
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        """
        Trains all base models and tunes ensemble blending weights & decision threshold
        using validation data to minimize total financial cost.
        """
        for name, model in self.models.items():
            if sample_weight is not None and hasattr(model, "fit"):
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

        self.is_fitted = True

        # Optimize ensemble weights & threshold using validation data if available
        if X_val is not None and y_val is not None:
            self._optimize_weights_and_threshold(X_val, y_val)
        else:
            self.optimal_threshold = self.cost_matrix.bayes_optimal_threshold

        return self

    def _optimize_weights_and_threshold(self, X_val, y_val):
        """
        Grid searches over ensemble weights simplex [w_lr, w_rf, w_xgb]
        and tunes probability threshold to achieve minimum financial loss.
        """
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict_proba(X_val)[:, 1]

        best_cost = float("inf")
        best_weights = self.weights
        best_threshold = self.optimal_threshold

        # Candidate weight triplets
        candidate_weights = [
            (0.1, 0.45, 0.45),
            (0.2, 0.40, 0.40),
            (0.33, 0.33, 0.34),
            (0.15, 0.50, 0.35),
            (0.15, 0.35, 0.50),
            (0.05, 0.40, 0.55),
            (0.05, 0.55, 0.40)
        ]

        threshold_scan = np.linspace(0.02, 0.95, 50)

        for w_lr, w_rf, w_xgb in candidate_weights:
            combined_prob = (
                w_lr * preds["Logistic Regression"] +
                w_rf * preds["Random Forest"] +
                w_xgb * preds["XGBoost"]
            )
            for th in threshold_scan:
                y_pred = (combined_prob >= th).astype(int)
                cost_res = self.cost_matrix.calculate_total_cost(y_val, y_pred)
                if cost_res["total_cost"] < best_cost:
                    best_cost = cost_res["total_cost"]
                    best_weights = {"Logistic Regression": w_lr, "Random Forest": w_rf, "XGBoost": w_xgb}
                    best_threshold = float(th)

        self.weights = best_weights
        self.optimal_threshold = best_threshold

    def predict_proba(self, X):
        """Returns weighted ensemble predicted probabilities [P(0), P(1)]."""
        if not self.is_fitted:
            raise ValueError("Ensemble has not been fitted!")

        prob_1 = (
            self.weights["Logistic Regression"] * self.lr_model.predict_proba(X)[:, 1] +
            self.weights["Random Forest"] * self.rf_model.predict_proba(X)[:, 1] +
            self.weights["XGBoost"] * self.xgb_model.predict_proba(X)[:, 1]
        )
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X, threshold=None):
        """Predicts binary class labels using the cost-optimal decision threshold."""
        th = threshold if threshold is not None else self.optimal_threshold
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= th).astype(int)

    def get_feature_importances(self, feature_names=None):
        """
        Aggregates normalized feature importances from tree base models (RF & XGBoost).
        """
        rf_imp = self.rf_model.feature_importances_
        xgb_imp = self.xgb_model.feature_importances_

        # Normalize
        rf_norm = rf_imp / np.sum(rf_imp) if np.sum(rf_imp) > 0 else rf_imp
        xgb_norm = xgb_imp / np.sum(xgb_imp) if np.sum(xgb_imp) > 0 else xgb_imp

        combined_imp = 0.5 * rf_norm + 0.5 * xgb_norm

        if feature_names is not None and len(feature_names) == len(combined_imp):
            return dict(sorted(zip(feature_names, combined_imp), key=lambda x: x[1], reverse=True))
        return combined_imp
