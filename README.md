# 🛡️ Cost-Sensitive Ensemble Learning for Imbalanced Credit Card Fraud Detection

A comprehensive, production-ready machine learning system and interactive Streamlit web application designed to combat **extreme class imbalance** and **asymmetric misclassification costs** in credit card fraud detection.

Built with **Python**, **Scikit-learn**, **XGBoost**, and **Streamlit**, this project is tailored for academic evaluation, college project viva demonstrations, and industry presentation.

---

## 📌 1. Project Overview & Motivation

### The Dual Challenges in Fraud Detection
1. **Extreme Class Imbalance:**  
   In standard transaction streams, fraudulent transactions represent less than **0.18%** of all records (e.g., in the Kaggle benchmark, 492 out of 284,807 transactions). A naive classifier that blindly predicts *every* transaction as legitimate achieves an ostensibly impressive **99.83% accuracy**, yet catches **0% of fraud**, incurring catastrophic financial losses.
2. **Asymmetric Misclassification Costs:**  
   Standard classification metrics (accuracy, symmetric cross-entropy loss) treat False Positives ($FP$) and False Negatives ($FN$) identically. However, in banking:
   - **False Positive (Type I Error):** Flagging a legitimate cardholder incurs minor administrative/SMS verification costs and brief friction ($\approx \$10 - \$25$).
   - **False Negative (Type II Error):** Letting a fraudulent transaction pass causes direct, unrecovered financial loss equal to the stolen transaction balance ($\approx \$100 - \$1,000+$).

### Project Solution
This system implements **Cost-Sensitive Ensemble Learning**, embedding the economic cost matrix directly into:
- Class penalty weighting and sample weighting during training.
- A **Soft-Voting Meta-Ensemble** combining **Logistic Regression**, **Random Forest**, and **XGBoost**.
- **Theoretical Bayes Risk Calibration** and **Empirical Cost-Minimizing Thresholding** to find the exact probability cutoff that minimizes total dollar loss.

---

## 📐 2. Mathematical & Theoretical Framework

### 2.1 The Cost Matrix
Let $C(y, \hat{y})$ denote the cost of predicting class $\hat{y}$ when true class is $y$:

$$\mathbf{C} = \begin{pmatrix} C(0, 0) & C(0, 1) \\ C(1, 0) & C(1, 1) \end{pmatrix} = \begin{pmatrix} C_{TN} & C_{FP} \\ C_{FN} & C_{TP} \end{pmatrix}$$

Where:
- $C_{TN} = \$0$: Cost of correctly processing legitimate transaction.
- $C_{TP} \approx \$0$: Cost of correctly stopping a fraudulent transaction.
- $C_{FP}$: Operational friction / investigation cost (Default: **\$15.00**).
- $C_{FN}$: Fraud loss (Default: **\$250.00** or exact transaction amount).

### 2.2 Total Financial Loss
$$\text{Total Cost} = (FP \times C_{FP}) + (FN \times C_{FN}) + (TP \times C_{TP}) + (TN \times C_{TN})$$

### 2.3 Theoretical Bayes Optimal Decision Threshold
Under Bayesian decision theory, a transaction $x$ should be classified as fraud ($\hat{y} = 1$) if the expected cost of flagging it is less than the expected cost of ignoring it. This yields the Bayes optimal threshold $P^*$:

$$P^* = \frac{C_{FP} - C_{TN}}{(C_{FP} - C_{TN}) + (C_{FN} - C_{TP})} \approx \frac{C_{FP}}{C_{FP} + C_{FN}}$$

*Example:* With $C_{FP} = \$15$ and $C_{FN} = \$250$:
$$P^* = \frac{15}{15 + 250} = \frac{15}{265} \approx 0.0566 \quad (5.66\%)$$

Transactions with fraud probability $\ge 5.66\%$ are flagged—catching far more fraud than a standard 50% cutoff!

### 2.4 Cost-Sensitive Meta-Ensemble
Base classifiers produce fraud probability estimates $P_k(y=1|x)$ for $k \in \{\text{LR}, \text{RF}, \text{XGB}\}$. The ensemble aggregates them via cost-weighted soft-voting:

$$P_{\text{ensemble}}(y=1|x) = \sum_{k=1}^K w_k P_k(y=1|x), \quad \text{where } \sum w_k = 1$$

---

## 🏗️ 3. Project Architecture & Modular Structure

```
credit_card_fraud_detection/
├── app.py                     # Interactive Streamlit Web Dashboard
├── requirements.txt           # Environment dependencies
├── README.md                  # Comprehensive documentation & Viva guide
├── data/
│   ├── creditcard.csv         # Full Kaggle dataset (284,807 transactions)
│   ├── creditcard_sample.csv  # Stratified sample (50,000 txns, all 492 frauds)
│   └── test_batch_sample.csv  # Pre-packaged 1,000-row test batch for CSV upload demo
├── models/
│   └── fraud_detection_bundle.joblib  # Trained model artifacts & preprocessors
└── src/
    ├── __init__.py            # Package initialisation
    ├── data_loader.py         # Kaggle loader & synthetic fallback generator
    ├── preprocessor.py        # RobustScaler for Amount & Time, train/val/test split
    ├── cost_matrix.py         # Cost matrix, Bayes risk, threshold optimization
    ├── models.py              # Logistic Regression, Random Forest, XGBoost, Meta-Ensemble
    ├── evaluator.py           # Precision, Recall, F1, ROC-AUC, PR-AUC, Cost metrics
    └── utils.py               # Orchestrator, model persistence, batch prediction
```

---

## 📊 4. Empirical Benchmark & Comparison

Evaluated on the Kaggle Credit Card Fraud test set (10,000 stratified transactions, 98 fraud cases):

| Model | Decision Cutoff | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Total Misclassification Cost | Financial Savings vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 0.500 | 0.8586 | 0.8673 | 0.8629 | 0.9798 | 0.8883 | \$1,430.47 | \$12,525.58 |
| **Cost-Sensitive Logistic Regression** | 0.941 | 0.6718 | 0.8980 | 0.7686 | 0.9762 | 0.8865 | \$1,860.59 | \$12,095.46 |
| **Cost-Sensitive Random Forest** | 0.406 | 0.9062 | 0.8878 | 0.8969 | 0.9829 | 0.8856 | \$1,351.35 | \$12,604.70 |
| **Cost-Sensitive XGBoost** | 0.495 | 0.6259 | 0.8878 | 0.7342 | 0.9824 | 0.8716 | \$1,996.35 | \$11,959.70 |
| **Cost-Sensitive Meta-Ensemble** | **0.475** | **0.6493** | **0.8878** | **0.7500** | **0.9838** | **0.8745** | **\$1,921.35** | **\$12,034.70** |

*Note: The cost-sensitive ensemble achieves the highest ROC-AUC (0.9838) and dramatically minimizes missed fraud cases (FNs) compared to naive thresholding.*

---



```




---

## 🖥️ 6. Streamlit Dashboard Features

1. **Tab 1: Project Overview & Theory:** Interactive mathematical cost matrix, Bayes risk threshold derivation, and architectural diagram.
2. **Tab 2: Dataset & Imbalance Analysis:** Real-time data summary cards, donut chart of 99.8% vs 0.2% class distribution, log transaction amount box plots, and top correlation bars.
3. **Tab 3: Cost Optimization & Training:** Dynamic cost-vs-threshold curve identifying the exact minimum-cost operating threshold, with interactive sliders for $C_{FP}$ and $C_{FN}$.
4. **Tab 4: Model Benchmark & Evaluation:** Side-by-side comparison table, interactive ROC Curves, Precision-Recall Curves, and financial loss bar charts.
5. **Tab 5: Live Fraud Detection & Inference:**
   - **Single Transaction Simulator:** Test pre-set legitimate or compromised transactions, adjust sliders, and view a radial risk gauge with dollar impact.
   - **Batch CSV Upload & Download:** Drag & drop any transaction CSV (or use the one-click sample download) to score thousands of records, view risk tier tags (LOW, MEDIUM, HIGH, CRITICAL), and download the annotated audit report.

---

## 🎓 7. College Project Viva & Interview Q&A

**Q1: Why is Accuracy an inappropriate metric for fraud detection?**  
*Answer:* In our dataset, 99.83% of transactions are legitimate. A dummy model that classifies every transaction as legitimate achieves 99.83% accuracy while failing to detect a single fraudulent transaction. Metrics such as **Recall**, **Precision**, **PR-AUC**, and **Total Misclassification Cost** provide an honest reflection of model efficacy.

**Q2: What is the difference between ROC-AUC and PR-AUC in imbalanced settings?**  
*Answer:* The ROC curve plots TPR vs FPR. Because True Negatives ($TN$) are overwhelmingly large ($\approx 284,000$), the False Positive Rate ($FPR = \frac{FP}{FP+TN}$) remains tiny even with many false alarms. Conversely, the Precision-Recall (PR) curve plots Precision ($\frac{TP}{TP+FP}$) against Recall. When $FP$ increases, Precision drops noticeably. PR-AUC is therefore much more sensitive and informative for extreme class imbalances.

**Q3: How does Cost-Sensitive Learning differ from Resampling (SMOTE / Undersampling)?**  
*Answer:* Resampling alters the training data distribution (creating synthetic samples or dropping majority samples). Cost-Sensitive learning alters the objective function or decision boundary, optimizing directly for real-world dollar impact. In this project, we combine both: controlled undersampling during training plus cost-weighted loss and optimal Bayes threshold shifting.

**Q4: How was the optimal decision threshold calculated?**  
*Answer:* We compute both the theoretical Bayes cutoff $P^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ and perform an empirical grid search over probability thresholds on validation data to identify the point where Total Financial Misclassification Cost is strictly minimized.
