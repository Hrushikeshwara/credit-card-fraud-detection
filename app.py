"""
Cost-Sensitive Ensemble Learning for Imbalanced Credit Card Fraud Detection
Streamlit Web Application
"""

import os
import sys
import tempfile

# Configure matplotlib temporary directory to avoid permissions issues in sandboxed environments
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure local src/ package is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import (
    load_credit_card_data,
    get_dataset_statistics,
    get_default_data_paths
)
from src.cost_matrix import CostMatrix
from src.utils import (
    load_model_artifacts,
    run_full_training_pipeline,
    predict_batch_transactions,
    get_preset_demo_transactions
)


# ==========================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Cost-Sensitive Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished college project demonstration
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .highlight-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .fraud-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid #F87171;
    }
    .legit-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid #86EFAC;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED DATA & ARTIFACTS HELPERS
# ==========================================
@st.cache_data(show_spinner=False)
def get_cached_dataset(use_sample=True, custom_file=None):
    if custom_file is not None:
        return pd.read_csv(custom_file)
    return load_credit_card_data(use_sample=use_sample)


def get_or_train_artifacts(cost_fp, cost_fn, force_retrain=False):
    """Retrieves saved model artifacts or triggers training if needed."""
    if not force_retrain:
        bundle = load_model_artifacts()
        if bundle is not None:
            # Check if cached costs match current configuration
            cached_cm = bundle.get("cost_matrix")
            if cached_cm and abs(cached_cm.cost_fp - cost_fp) < 1e-3 and abs(cached_cm.cost_fn - cost_fn) < 1e-3:
                return bundle

    # Train fresh models
    df = get_cached_dataset(use_sample=True)
    with st.spinner("Training Cost-Sensitive Models (Logistic Regression, Random Forest, XGBoost, and Meta-Ensemble)..."):
        bundle = run_full_training_pipeline(
            df=df,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            use_undersampling=True,
            legit_to_fraud_ratio=15
        )
    return bundle


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("https://img.icons8.com/fluency/96/bank-cards.png", width=70)
st.sidebar.title("Configuration & Parameters")

st.sidebar.markdown("### 1. Cost Matrix Configuration")
st.sidebar.caption("Define the financial penalty for misclassification:")

cost_fp = st.sidebar.slider(
    "False Positive Cost ($C_{FP}$)",
    min_value=1.0,
    max_value=100.0,
    value=15.0,
    step=1.0,
    help="Cost of false alert: investigation labor, user SMS verification, and customer friction."
)

cost_fn = st.sidebar.slider(
    "False Negative Cost ($C_{FN}$)",
    min_value=20.0,
    max_value=1000.0,
    value=250.0,
    step=10.0,
    help="Cost of missed fraud: full unrecovered fraudulent transaction value."
)

cost_ratio = cost_fn / max(1e-3, cost_fp)
cost_matrix = CostMatrix(cost_fp=cost_fp, cost_fn=cost_fn)
bayes_th = cost_matrix.bayes_optimal_threshold

st.sidebar.markdown(f"""
<div style="background-color: #F1F5F9; padding: 10px; border-radius: 6px; border-left: 4px solid #3B82F6; font-size: 0.88rem;">
    <b>Asymmetry Ratio (FN : FP):</b> {cost_ratio:.1f}x<br>
    <b>Theoretical Bayes Threshold:</b> <code style="font-weight: bold; color: #1E3A8A;">{bayes_th:.4f}</code><br>
    <i>Transactions with P(Fraud) &ge; {bayes_th:.3f} are financially optimal to flag.</i>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Dataset Selection")
data_mode = st.sidebar.radio(
    "Select Training Dataset:",
    ["Stratified Sample (50,000 txns, all 492 frauds)", "Full Kaggle Dataset (284,807 txns)", "Upload Custom CSV"],
    index=0
)

uploaded_custom_file = None
if data_mode == "Upload Custom CSV":
    uploaded_custom_file = st.sidebar.file_uploader("Upload CSV (Time, V1..V28, Amount, Class)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Model Management")
retrain_clicked = st.sidebar.button("🚀 Retrain Models with New Costs", use_container_width=True)

# Load / Train Artifacts
if data_mode == "Upload Custom CSV" and uploaded_custom_file is not None:
    df_active = pd.read_csv(uploaded_custom_file)
    artifacts = run_full_training_pipeline(df_active, cost_fp=cost_fp, cost_fn=cost_fn)
elif data_mode == "Full Kaggle Dataset (284,807 txns)":
    paths = get_default_data_paths()
    if os.path.exists(paths["full"]):
        df_active = pd.read_csv(paths["full"])
    else:
        df_active = get_cached_dataset(use_sample=True)
    if retrain_clicked:
        artifacts = run_full_training_pipeline(df_active, cost_fp=cost_fp, cost_fn=cost_fn)
    else:
        artifacts = get_or_train_artifacts(cost_fp, cost_fn, force_retrain=False)
else:
    df_active = get_cached_dataset(use_sample=True)
    if retrain_clicked:
        artifacts = get_or_train_artifacts(cost_fp, cost_fn, force_retrain=True)
    else:
        artifacts = get_or_train_artifacts(cost_fp, cost_fn, force_retrain=False)


# ==========================================
# MAIN INTERFACE HEADER
# ==========================================
st.markdown('<div class="main-header">🛡️ Cost-Sensitive Ensemble Learning for Imbalanced Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-header">
    An End-to-End Machine Learning System integrating <b>Class Imbalance Mitigation</b>, 
    <b>Bayes Risk Cost Optimization</b>, and a <b>Soft-Voting Meta-Ensemble</b> (Logistic Regression, Random Forest, XGBoost) 
    on the Kaggle Credit Card Fraud Dataset.
</div>
""", unsafe_allow_html=True)


# ==========================================
# NAVIGATION TABS
# ==========================================
tab_overview, tab_eda, tab_train, tab_eval, tab_predict = st.tabs([
    "📖 Project Overview & Theory",
    "📊 Dataset & Imbalance Analysis",
    "⚙️ Cost Optimization & Training",
    "📈 Model Benchmark & Evaluation",
    "🔍 Live Fraud Detection & Batch Audit"
])


# ==========================================
# TAB 1: PROJECT OVERVIEW & THEORY
# ==========================================
with tab_overview:
    st.markdown("### 🎯 Executive Summary & Problem Formulation")
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
        In real-world financial fraud detection, two profound challenges render standard machine learning algorithms ineffective:

        1. **Extreme Class Imbalance:** Fraudulent transactions typically represent less than **0.2%** of all network transactions. 
           A trivial classifier that naively predicts every transaction as *Legitimate* achieves **99.83% accuracy**, yet allows 100% of fraud losses to occur unchecked.
        2. **Asymmetric Misclassification Costs:** Traditional models optimize for standard accuracy or symmetric 0-1 loss, treating a False Positive (FP) and a False Negative (FN) identically. 
           In banking operations:
           - **False Positive (FP):** Inconveniences a legitimate cardholder with an SMS or call (cost $\\approx \\$10 - \\$25$).
           - **False Negative (FN):** Results in direct, unrecovered financial loss equal to the full stolen transaction value (cost $\\approx \\$100 - \\$1,000+$).
        """)

        st.info("""
        **Project Innovation:** This project develops a **Cost-Sensitive Ensemble Learning Architecture** that directly incorporates 
        the financial cost matrix into both training sample weights, class penalties, and post-hoc Bayes decision threshold calibration.
        """)

    with col2:
        st.markdown("#### 📐 Cost Matrix Formulation")
        st.latex(r"""
        C = \begin{pmatrix}
        C(TN) & C(FP) \\
        C(FN) & C(TP)
        \end{pmatrix} = \begin{pmatrix}
        \$0 & C_{FP} \\
        C_{FN} & \$0
        \end{pmatrix}
        """)
        st.markdown(f"""
        - **Total Financial Loss:**
        $$Cost_{{total}} = (FP \\times C_{{FP}}) + (FN \\times C_{{FN}})$$
        - **Theoretical Bayes Optimal Threshold:**
        $$P^* = \\frac{{C_{{FP}} - C_{{TN}}}}{{(C_{{FP}} - C_{{TN}}) + (C_{{FN}} - C_{{TP}})}} = \\frac{{C_{{FP}}}}{{C_{{FP}} + C_{{FN}}}}$$
        - Current Threshold: **{bayes_th:.4f}** (vs Default 0.50)
        """)

    st.markdown("---")
    st.markdown("### 🏗️ End-to-End System Architecture")

    st.markdown("""
    ```
    ┌──────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
    │  Kaggle Credit Card  │  ───> │ RobustScaler (Amount,  │  ───> │  Imbalance Handling   │
    │  Transactions Data   │       │ Time) + Scaled PCA V1-28│       │ (Cost-Weights + Strat) │
    └──────────────────────┘       └────────────────────────┘       └────────────────────────┘
                                                                                 │
                                                                                 ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                       TRI-MODEL BASE CLASSIFIERS (Cost-Weighted)                       │
    │   1. Cost-Sensitive Logistic Regression   2. Balanced Random Forest   3. Cost-Sensitive XGBoost │
    └────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                   META-ENSEMBLE SOFT-VOTING & BAYES RISK CALIBRATION                   │
    │      P_ensemble = w_LR * P_LR + w_RF * P_RF + w_XGB * P_XGB   |   Threshold P* Tuning  │
    └────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                     PRODUCTION INTERFERENCE & FINANCIAL AUDIT                          │
    │          • Single Transaction Risk Scoring   • Batch CSV Audit & Cost Savings          │
    └────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    """)


# ==========================================
# TAB 2: DATASET & IMBALANCE ANALYSIS
# ==========================================
with tab_eda:
    st.markdown("### 📊 Dataset Exploration & Class Imbalance Analysis")

    stats = get_dataset_statistics(df_active)

    # Metric Cards Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", f"{stats['total_records']:,}")
    m2.metric("Legitimate Txns", f"{stats['legitimate_count']:,}")
    m3.metric("Fraudulent Txns", f"{stats['fraud_count']:,}")
    m4.metric("Fraud Proportion", f"{stats['fraud_percentage']:.3f}%")
    m5.metric("Class Ratio", stats["imbalance_ratio"])

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Class Distribution (Severe Imbalance)")
        pie_fig = go.Figure(data=[go.Pie(
            labels=["Legitimate (0)", "Fraudulent (1)"],
            values=[stats["legitimate_count"], stats["fraud_count"]],
            hole=0.45,
            marker_colors=["#10B981", "#EF4444"],
            textinfo="label+percent",
            insidetextorientation="radial"
        )])
        pie_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_chart2:
        st.markdown("#### Transaction Amounts: Legitimate vs Fraud")
        amt_df = df_active[["Amount", "Class"]].copy()
        amt_df["Type"] = np.where(amt_df["Class"] == 1, "Fraud", "Legitimate")
        amt_df["Log_Amount"] = np.log10(amt_df["Amount"].clip(lower=0.1))

        box_fig = px.box(
            amt_df,
            x="Type",
            y="Log_Amount",
            color="Type",
            color_discrete_map={"Legitimate": "#10B981", "Fraud": "#EF4444"},
            points="outliers",
            labels={"Log_Amount": "Log10(Amount in $)"}
        )
        box_fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(box_fig, use_container_width=True)

    st.markdown("#### Top Feature Correlations with Fraud")
    # Compute correlation with Class
    corr_series = df_active.corr()["Class"].drop("Class").sort_values()
    top_neg = corr_series.head(5)
    top_pos = corr_series.tail(5)
    top_corr = pd.concat([top_neg, top_pos])

    fig_corr = px.bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation="h",
        color=top_corr.values,
        color_continuous_scale="RdBu_r",
        labels={"x": "Pearson Correlation with Class", "y": "Feature"},
        title="Key Predictive PCA Features (Notice Strong Negative Shifts in V14, V12, V10, V17)"
    )
    fig_corr.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("🔍 Preview Raw Dataset Sample (First 10 Rows)"):
        st.dataframe(df_active.head(10), use_container_width=True)


# ==========================================
# TAB 3: COST OPTIMIZATION & TRAINING
# ==========================================
with tab_train:
    st.markdown("### ⚙️ Cost-Sensitive Threshold Optimization")
    st.markdown("""
    Unlike standard classifiers that apply an arbitrary `0.5` decision cutoff, 
    our cost-sensitive framework scans the entire probability spectrum to identify the 
    **Empirical Minimum Cost Threshold** that minimizes financial loss for your organization.
    """)

    best_eval = artifacts["evaluations"]["Cost-Sensitive Ensemble"]
    curve_data = best_eval["threshold_curve"]

    col_plot, col_summary = st.columns([2, 1])

    with col_plot:
        # Plot Cost vs Threshold Curve
        th_df = pd.DataFrame({
            "Threshold": curve_data["thresholds"],
            "Total_Cost": curve_data["costs"],
            "FP_Count": curve_data["fp_counts"],
            "FN_Count": curve_data["fn_counts"]
        })

        cost_fig = go.Figure()
        cost_fig.add_trace(go.Scatter(
            x=th_df["Threshold"],
            y=th_df["Total_Cost"],
            mode="lines",
            name="Total Financial Cost ($)",
            line=dict(color="#1E3A8A", width=3)
        ))

        # Vertical line at Optimal Threshold
        opt_th = curve_data["optimal_threshold"]
        min_cost = curve_data["minimum_cost"]
        cost_fig.add_vline(
            x=opt_th,
            line_dash="dash",
            line_color="#10B981",
            annotation_text=f"Optimal Threshold: {opt_th:.3f}",
            annotation_position="top left"
        )

        # Standard 0.5 threshold line
        cost_fig.add_vline(
            x=0.5,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="Standard (0.50)",
            annotation_position="top right"
        )

        cost_fig.update_layout(
            title="Total Financial Cost vs. Classification Threshold",
            xaxis_title="Decision Probability Threshold",
            yaxis_title="Total Financial Misclassification Cost ($)",
            height=420,
            hovermode="x unified"
        )
        st.plotly_chart(cost_fig, use_container_width=True)

    with col_summary:
        st.markdown("#### Cost Optimization Impact")
        st.metric(
            label="Optimal Cost Threshold (T*)",
            value=f"{opt_th:.3f}",
            delta=f"{opt_th - 0.5:.3f} vs standard 0.5"
        )
        st.metric(
            label="Total Cost at T*",
            value=f"${min_cost:,.2f}"
        )
        st.metric(
            label="Cost at Standard 0.5 Threshold",
            value=f"${curve_data['standard_cost']:,.2f}"
        )
        st.metric(
            label="Financial Savings vs No Model",
            value=f"${curve_data['savings_vs_baseline']:,.2f}",
            delta="Money Saved",
            delta_color="normal"
        )

        st.markdown("#### Ensemble Model Blending Weights")
        weights = artifacts["best_model"].weights
        for model_name, w in weights.items():
            st.progress(float(w), text=f"{model_name}: {w*100:.1f}%")


# ==========================================
# TAB 4: MODEL BENCHMARK & EVALUATION
# ==========================================
with tab_eval:
    st.markdown("### 📈 Comprehensive Model Benchmark & Evaluation")
    st.markdown("Comparative performance metrics across standard and cost-sensitive configurations:")

    comp_df = artifacts["comparison_df"]
    st.dataframe(comp_df, use_container_width=True)

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC Curve)")
        roc_fig = go.Figure()
        for name, ev in artifacts["evaluations"].items():
            roc_fig.add_trace(go.Scatter(
                x=ev["roc_curve"]["fpr"],
                y=ev["roc_curve"]["tpr"],
                mode="lines",
                name=f"{name} (AUC={ev['roc_auc']:.3f})"
            ))
        # Diagonal reference line
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Random Guess"
        ))
        roc_fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            height=380,
            legend=dict(font=dict(size=9))
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with col_pr:
        st.markdown("#### Precision-Recall Curve (PR-AUC)")
        pr_fig = go.Figure()
        for name, ev in artifacts["evaluations"].items():
            pr_fig.add_trace(go.Scatter(
                x=ev["pr_curve"]["recall"],
                y=ev["pr_curve"]["precision"],
                mode="lines",
                name=f"{name} (PR-AUC={ev['pr_auc']:.3f})"
            ))
        pr_fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=380,
            legend=dict(font=dict(size=9))
        )
        st.plotly_chart(pr_fig, use_container_width=True)

    col_cost_bar, col_feat_imp = st.columns(2)

    with col_cost_bar:
        st.markdown("#### Total Misclassification Cost Comparison ($)")
        costs_list = [ev["total_cost"] for ev in artifacts["evaluations"].values()]
        models_list = list(artifacts["evaluations"].keys())

        bar_cost_fig = px.bar(
            x=models_list,
            y=costs_list,
            color=costs_list,
            color_continuous_scale="Viridis_r",
            labels={"x": "Model", "y": "Total Financial Cost ($)"},
            title="Total Financial Loss (Lower is Better)"
        )
        bar_cost_fig.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(bar_cost_fig, use_container_width=True)

    with col_feat_imp:
        st.markdown("#### Ensemble Feature Importances (Top 10)")
        imp_dict = artifacts["best_model"].get_feature_importances(
            artifacts["preprocessor"].feature_names
        )
        if isinstance(imp_dict, dict):
            top_features = list(imp_dict.keys())[:10]
            top_weights = [imp_dict[k] for k in top_features]
            fig_imp = px.bar(
                x=top_weights,
                y=top_features,
                orientation="h",
                labels={"x": "Importance Weight", "y": "Feature"},
                title="Top Predictive Features in Ensemble"
            )
            fig_imp.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# TAB 5: LIVE FRAUD DETECTION & INFERENCE
# ==========================================
with tab_predict:
    st.markdown("### 🔍 Live Fraud Inference & Batch Audit Engine")

    subtab_single, subtab_batch = st.tabs(["Single Transaction Simulator", "Batch CSV Upload & Audit"])

    # ----------------------------------------------------
    # SUBTAB: SINGLE TRANSACTION
    # ----------------------------------------------------
    with subtab_single:
        st.markdown("#### Simulate and Analyze a Credit Card Transaction")

        demo_presets = get_preset_demo_transactions()
        chosen_preset = st.selectbox(
            "Load Transaction Template:",
            ["Custom Input"] + list(demo_presets.keys())
        )

        preset_vals = demo_presets.get(chosen_preset, {})

        c1, c2, c3 = st.columns(3)
        with c1:
            input_amount = st.number_input(
                "Transaction Amount ($):",
                min_value=0.01,
                max_value=25000.0,
                value=float(preset_vals.get("Amount", 85.00)),
                step=5.0
            )
            input_time = st.number_input(
                "Transaction Time (seconds from first txn):",
                min_value=0.0,
                max_value=200000.0,
                value=float(preset_vals.get("Time", 50000.0)),
                step=100.0
            )

        with c2:
            st.caption("Key Anomaly Indicators (Top Fraud Correlates):")
            input_v14 = st.slider("V14 (Primary indicator):", -15.0, 10.0, float(preset_vals.get("V14", -0.2)), 0.1)
            input_v12 = st.slider("V12:", -15.0, 10.0, float(preset_vals.get("V12", 0.1)), 0.1)
            input_v10 = st.slider("V10:", -15.0, 10.0, float(preset_vals.get("V10", -0.1)), 0.1)

        with c3:
            input_v17 = st.slider("V17:", -15.0, 10.0, float(preset_vals.get("V17", 0.0)), 0.1)
            input_v4 = st.slider("V4:", -10.0, 15.0, float(preset_vals.get("V4", 0.0)), 0.1)
            input_v11 = st.slider("V11:", -10.0, 15.0, float(preset_vals.get("V11", 0.0)), 0.1)

        # Assemble transaction vector
        single_txn_dict = {f"V{i}": float(preset_vals.get(f"V{i}", 0.0)) for i in range(1, 29)}
        single_txn_dict["Amount"] = input_amount
        single_txn_dict["Time"] = input_time
        single_txn_dict["V14"] = input_v14
        single_txn_dict["V12"] = input_v12
        single_txn_dict["V10"] = input_v10
        single_txn_dict["V17"] = input_v17
        single_txn_dict["V4"] = input_v4
        single_txn_dict["V11"] = input_v11

        if st.button("⚡ Score Transaction with Ensemble", use_container_width=True):
            preprocessor = artifacts["preprocessor"]
            best_model = artifacts["best_model"]

            # Vectorize
            X_single = preprocessor.transform_single_transaction(single_txn_dict)
            fraud_prob = float(best_model.predict_proba(X_single)[0, 1])
            is_fraud = fraud_prob >= best_model.optimal_threshold

            res_c1, res_c2 = st.columns([1, 1.2])

            with res_c1:
                # Radial Gauge
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Fraud Probability Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#EF4444" if is_fraud else "#10B981"},
                        "steps": [
                            {"range": [0, best_model.optimal_threshold * 100], "color": "#DCFCE7"},
                            {"range": [best_model.optimal_threshold * 100, 100], "color": "#FEE2E2"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": best_model.optimal_threshold * 100
                        }
                    }
                ))
                gauge_fig.update_layout(height=260, margin=dict(t=40, b=10, l=30, r=30))
                st.plotly_chart(gauge_fig, use_container_width=True)

            with res_c2:
                st.markdown("#### Classification Decision & Financial Impact")
                if is_fraud:
                    st.markdown("""
                    <div style="background-color: #FEF2F2; border: 2px solid #EF4444; border-radius: 8px; padding: 15px;">
                        <span class="fraud-badge">🚨 FRAUD DETECTED - TRANSACTION BLOCKED</span>
                        <p style="margin-top: 10px; color: #7F1D1D; font-size: 0.95rem;">
                            <b>Action:</b> Transaction flagged for immediate secondary authorization (SMS/OTP verification or call).
                            <br><b>Prevented Fraud Loss:</b> ${:,.2f}
                            <br><b>Operational Friction Cost (FP Risk):</b> ${:,.2f}
                        </p>
                    </div>
                    """.format(input_amount, cost_fp), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 8px; padding: 15px;">
                        <span class="legit-badge">✅ TRANSACTION APPROVED - LEGITIMATE</span>
                        <p style="margin-top: 10px; color: #14532D; font-size: 0.95rem;">
                            <b>Action:</b> Transaction permitted without customer disruption.
                            <br><b>Risk Assessment:</b> Probability ({:.2f}%) is below cost-optimal cutoff ({:.2f}%).
                        </p>
                    </div>
                    """.format(fraud_prob * 100, best_model.optimal_threshold * 100), unsafe_allow_html=True)

    # ----------------------------------------------------
    # SUBTAB: BATCH CSV PROCESSING
    # ----------------------------------------------------
    with subtab_batch:
        st.markdown("#### Upload Transaction Batch for High-Throughput Audit")

        paths = get_default_data_paths()
        if os.path.exists(paths["test_batch"]):
            with open(paths["test_batch"], "rb") as f:
                st.download_button(
                    label="📥 Download Sample Batch File for Testing (1,000 real transactions)",
                    data=f,
                    file_name="creditcard_sample_test_batch.csv",
                    mime="text/csv",
                    help="Click to download a prepared batch of 1,000 Kaggle transactions ready to upload."
                )

        uploaded_batch = st.file_uploader(
            "Upload Transaction CSV for Batch Scoring:",
            type=["csv"],
            key="batch_uploader"
        )

        if uploaded_batch is not None:
            df_batch = pd.read_csv(uploaded_batch)
            st.success(f"Loaded {len(df_batch):,} transactions from `{uploaded_batch.name}`.")

            with st.spinner("Scoring batch transactions using Cost-Sensitive Ensemble..."):
                results_df = predict_batch_transactions(
                    df_batch,
                    artifacts["best_model"],
                    artifacts["preprocessor"]
                )

            # Summary KPIs
            n_total = len(results_df)
            n_fraud_flagged = int((results_df["Predicted_Label"] == 1).sum())
            n_legit_flagged = n_total - n_fraud_flagged

            total_fraud_amt_flagged = float(results_df[results_df["Predicted_Label"] == 1]["Amount"].sum()) if "Amount" in results_df.columns else 0.0

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Transactions Scored", f"{n_total:,}")
            b2.metric("Flagged as Fraud", f"{n_fraud_flagged:,}", delta=f"{(n_fraud_flagged/n_total)*100:.2f}% flag rate")
            b3.metric("Passed as Legitimate", f"{n_legit_flagged:,}")
            b4.metric("Flagged Fraud Value", f"${total_fraud_amt_flagged:,.2f}")

            # Risk distribution chart
            st.markdown("#### Risk Level Distribution")
            risk_counts = results_df["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Level", "Count"]
            risk_fig = px.bar(
                risk_counts,
                x="Risk_Level",
                y="Count",
                color="Risk_Level",
                color_discrete_map={
                    "LOW": "#10B981",
                    "MEDIUM": "#F59E0B",
                    "HIGH": "#F97316",
                    "CRITICAL": "#EF4444"
                }
            )
            risk_fig.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(risk_fig, use_container_width=True)

            st.markdown("#### Scored Transactions Table")
            display_cols = ["Time", "Amount", "Fraud_Probability", "Prediction", "Risk_Level"]
            available_cols = [c for c in display_cols if c in results_df.columns]
            st.dataframe(results_df[available_cols + [c for c in results_df.columns if c not in available_cols]].head(50), use_container_width=True)

            # Download Annotated Results
            csv_export = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Scored Batch CSV",
                data=csv_export,
                file_name=f"scored_transactions_{uploaded_batch.name}",
                mime="text/csv",
                use_container_width=True
            )


# ==========================================
# FOOTER & PROJECT METADATA
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding-bottom: 2rem;">
    <b>College Project Demonstration:</b> Cost-Sensitive Ensemble Learning for Imbalanced Credit Card Fraud Detection<br>
    Built with Python, Scikit-Learn, XGBoost, and Streamlit.
</div>
""", unsafe_allow_html=True)
