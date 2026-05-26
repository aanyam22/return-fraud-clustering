# =============================================================================
# app/dashboard.py
# Streamlit dashboard — two tabs:
#   1. Cluster Explorer  — scatter plot + profile table + distribution charts
#   2. Order Risk Scorer — manual inputs → pre-shipment return probability
#
# Run from project root:
#   streamlit run app/dashboard.py
# =============================================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.features import (
    CLUSTER_NAMES, CLUSTER_COLORS, ALL_COLS, TXN_COLS,
    PAYMENT_TYPE_ENC, encode_single_order
)

st.set_page_config(
    page_title="Return Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

# =============================================================================
# LOAD DATA & MODELS
# =============================================================================
@st.cache_data
def load_data():
    feats = pd.read_parquet("data/features/customer_features_clustered.parquet")
    feats["cluster_name"] = feats["cluster_hier"].map(CLUSTER_NAMES)
    return feats

@st.cache_resource
def load_models():
    model_A = joblib.load("outputs/model_A.pkl")
    model_B = joblib.load("outputs/model_B.pkl")
    return model_A, model_B

feats = load_data()
model_A, model_B = load_models()

# Derived stats for scorer defaults
overall_return_rate  = feats["return_rate"].mean()
overall_avg_value    = feats["avg_order_value"].mean()
overall_avg_days     = feats["avg_days_to_review"].mean()

# =============================================================================
# HEADER
# =============================================================================
st.title("🔍 Return Fraud Detection")
st.caption(
    "Olist Brazilian E-Commerce  ·  "
    "Unsupervised clustering (PCA + Ward + DBSCAN)  ·  "
    "Pre-shipment risk scoring (XGBoost)"
)
st.divider()

tab1, tab2 = st.tabs(["📊 Cluster Explorer", "⚠️ Order Risk Scorer"])

# =============================================================================
# TAB 1 — CLUSTER EXPLORER
# =============================================================================
with tab1:

    # ── Top KPI cards ─────────────────────────────────────────────────────────
    kpi_cols = st.columns(5)
    cluster_order = [
        "Fraud / Serial Abusers",
        "Policy Abusers",
        "High-Value Fast Returners",
        "Occasional Dissatisfied",
        "Loyal Low-Risk",
    ]
    for col, name in zip(kpi_cols, cluster_order):
        grp = feats[feats["cluster_name"] == name]
        if len(grp) == 0:
            continue
        col.metric(
            label=name,
            value=f"{len(grp):,} customers",
            delta=f"{grp['return_rate'].mean()*100:.1f}% return rate",
            delta_color="inverse" if "Loyal" in name else "normal",
        )

    st.divider()

    # ── Scatter plot ───────────────────────────────────────────────────────────
    st.subheader("Customer clusters: return rate vs order value")

    scatter_sample = feats.sample(min(3000, len(feats)), random_state=42)
    fig_scatter = px.scatter(
        scatter_sample,
        x="avg_order_value",
        y="return_rate",
        color="cluster_name",
        color_discrete_map=CLUSTER_COLORS,
        size="total_orders",
        size_max=14,
        opacity=0.65,
        hover_data={
            "avg_days_to_review": ":.1f",
            "total_orders": True,
            "dbscan_noise": True,
        },
        labels={
            "avg_order_value": "Avg order value (R$)",
            "return_rate":     "Return rate",
            "cluster_name":    "Cluster",
        },
    )
    fig_scatter.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Profile table ──────────────────────────────────────────────────────────
    st.subheader("Cluster profile summary")

    profile = (feats
        .groupby("cluster_name")
        .agg(
            customers        = ("customer_unique_id", "count"),
            return_rate      = ("return_rate",        "mean"),
            avg_order_value  = ("avg_order_value",    "mean"),
            avg_days_review  = ("avg_days_to_review", "mean"),
            pct_dbscan_noise = ("dbscan_noise",       "mean"),
        )
        .reset_index()
        .sort_values("return_rate", ascending=False)
    )
    profile["return_rate"]      = (profile["return_rate"]      * 100).round(1).astype(str) + "%"
    profile["avg_order_value"]  = "R$" + profile["avg_order_value"].round(0).astype(int).astype(str)
    profile["avg_days_review"]  = profile["avg_days_review"].round(1).astype(str) + "d"
    profile["pct_dbscan_noise"] = (profile["pct_dbscan_noise"] * 100).round(1).astype(str) + "%"
    profile.columns = ["Cluster", "Customers", "Return rate", "Avg order value",
                        "Avg days to review", "% DBSCAN noise"]
    st.dataframe(profile, use_container_width=True, hide_index=True)

    # ── Distribution charts ────────────────────────────────────────────────────
    st.subheader("Feature distributions by cluster")
    dist_col1, dist_col2, dist_col3 = st.columns(3)

    for col, feat, label in [
        (dist_col1, "return_rate",        "Return rate"),
        (dist_col2, "avg_order_value",    "Avg order value (R$)"),
        (dist_col3, "avg_days_to_review", "Avg days to review"),
    ]:
        fig = px.box(
            feats,
            x="cluster_name",
            y=feat,
            color="cluster_name",
            color_discrete_map=CLUSTER_COLORS,
            points=False,
            labels={"cluster_name": "", feat: label},
        )
        fig.update_layout(height=320, showlegend=False, margin=dict(t=10, b=60))
        fig.update_xaxes(tickangle=25)
        col.plotly_chart(fig, use_container_width=True)

    # ── DBSCAN noise highlight ─────────────────────────────────────────────────
    st.subheader("DBSCAN noise customers (highest anomaly risk)")
    noise = feats[feats["dbscan_noise"] == 1].sort_values("return_rate", ascending=False)
    st.caption(
        f"{len(noise):,} customers ({len(noise)/len(feats)*100:.1f}%) flagged as outliers by DBSCAN. "
        "These fall outside all normal behavioral clusters — highest fraud investigation priority."
    )
    st.dataframe(
        noise[["customer_unique_id", "cluster_name", "return_rate",
               "avg_order_value", "avg_days_to_review", "total_orders"]]
        .head(50)
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# TAB 2 — ORDER RISK SCORER
# =============================================================================
with tab2:

    st.subheader("Score a new order before shipment")
    st.caption(
        "Enter order and customer details to get a pre-shipment return probability. "
        "Model B (cluster-augmented) is used — AUC-PR 0.263 vs baseline 0.233."
    )

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("scorer_form"):
        st.markdown("**Order details**")
        c1, c2, c3 = st.columns(3)

        order_value    = c1.number_input("Order value (R$)",        min_value=1.0,  max_value=10000.0, value=150.0, step=10.0)
        payment_type   = c2.selectbox("Payment type",               options=list(PAYMENT_TYPE_ENC.keys())[:-1])
        seller_rr      = c3.slider("Seller return rate",            0.0, 1.0, 0.10, 0.01,
                                    help="Historical return rate of the seller fulfilling this order")
        category_rr    = c1.slider("Category return rate",          0.0, 1.0, 0.10, 0.01,
                                    help="Historical return rate of this product category")

        st.markdown("**Customer behavioral profile**")
        d1, d2, d3 = st.columns(3)

        tenure_days    = d1.number_input("Customer tenure (days)",  min_value=0,    max_value=2000,    value=180)
        cust_rr        = d2.slider("Customer return rate",          0.0, 1.0, float(round(overall_return_rate, 2)), 0.01)
        avg_val        = d3.number_input("Customer avg order value (R$)", min_value=1.0, max_value=5000.0,
                                          value=float(round(overall_avg_value, 0)))
        avg_days       = d1.slider("Avg days to review",            0.0, 90.0, float(round(overall_avg_days, 1)), 0.5)
        min_days       = d2.slider("Min days to review (fastest complaint)", 0.0, 90.0, float(round(overall_avg_days, 1)), 0.5)

        st.markdown("**Cluster assignment**")
        e1, e2, e3 = st.columns(3)

        cluster_hier   = e1.selectbox(
            "Customer cluster",
            options=list(CLUSTER_NAMES.keys()),
            format_func=lambda k: f"{k} — {CLUSTER_NAMES[k]}",
            index=4,  # default: Loyal Low-Risk
        )
        dbscan_noise   = e2.selectbox("DBSCAN noise flag", options=[0, 1],
                                       format_func=lambda x: "Yes — anomalous" if x else "No — normal",
                                       help="Was this customer flagged as a DBSCAN outlier?")
        cluster_dbscan = e3.number_input("DBSCAN cluster label (-1 = noise)", min_value=-1, max_value=10, value=0)

        st.markdown("**Distance to cluster centroids**")
        f_cols = st.columns(5)
        dist_defaults = [1.5, 1.5, 1.5, 1.5, 1.5]
        dist_to_clusters = [
            f_cols[i].number_input(
                f"Dist to cluster {i+1}\n({CLUSTER_NAMES[i+1][:12]}...)",
                min_value=0.0, max_value=20.0,
                value=dist_defaults[i], step=0.1
            )
            for i in range(5)
        ]

        submitted = st.form_submit_button("Calculate return risk", type="primary", use_container_width=True)

    # ── Results ────────────────────────────────────────────────────────────────
    if submitted:
        X_single = encode_single_order(
            order_value=order_value,
            payment_type=payment_type,
            tenure_days=tenure_days,
            seller_return_rate=seller_rr,
            category_return_rate=category_rr,
            customer_return_rate=cust_rr,
            avg_order_value=avg_val,
            avg_days_to_review=avg_days,
            min_days_to_review=min_days,
            cluster_hier=cluster_hier,
            dbscan_noise=dbscan_noise,
            cluster_dbscan=cluster_dbscan,
            dist_to_clusters=dist_to_clusters,
        )

        # Both model scores for comparison
        score_A = model_A.predict_proba(X_single[TXN_COLS])[:, 1][0]
        score_B = model_B.predict_proba(X_single)[:, 1][0]

        st.divider()
        r1, r2, r3 = st.columns(3)

        r1.metric("Cluster-augmented score (Model B)", f"{score_B*100:.1f}%",
                   help="Primary model — uses behavioral cluster features")
        r2.metric("Baseline score (Model A)",          f"{score_A*100:.1f}%",
                   help="Transaction-only model — no cluster features")
        r3.metric("Cluster signal lift",
                   f"+{(score_B - score_A)*100:+.1f}pp",
                   help="How much the cluster features changed the prediction")

        # Risk band
        THRESHOLD = 0.62  # best F1 threshold from 05_evaluation.py
        if score_B >= THRESHOLD:
            st.error(
                f"🚨 **HIGH RISK** (score {score_B*100:.1f}% ≥ {THRESHOLD*100:.0f}% threshold)  \n"
                "Recommended actions: require signature on delivery, "
                "restrict to credit card only, flag for manual review before dispatch."
            )
        elif score_B >= 0.35:
            st.warning(
                f"⚠️ **MEDIUM RISK** (score {score_B*100:.1f}%)  \n"
                "Recommended actions: send post-delivery satisfaction check, "
                "log for pattern monitoring."
            )
        else:
            st.success(
                f"✅ **LOW RISK** (score {score_B*100:.1f}%)  \n"
                "No intervention needed. Process order normally."
            )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(score_B * 100, 1),
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e05c5c" if score_B >= THRESHOLD
                                  else "#f5a623" if score_B >= 0.35
                                  else "#3ecf8e"},
                "steps": [
                    {"range": [0,   35],  "color": "#eafaf1"},
                    {"range": [35,  62],  "color": "#fef9e7"},
                    {"range": [62,  100], "color": "#fdedec"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": THRESHOLD * 100,
                },
            },
            title={"text": "Return probability (Model B)"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Show which cluster was selected and its risk profile
        cluster_name = CLUSTER_NAMES[cluster_hier]
        grp = feats[feats["cluster_name"] == cluster_name]
        if len(grp) > 0:
            st.info(
                f"**Customer cluster: {cluster_name}**  \n"
                f"Cluster avg return rate: {grp['return_rate'].mean()*100:.1f}%  ·  "
                f"Avg order value: R${grp['avg_order_value'].mean():.0f}  ·  "
                f"Customers in cluster: {len(grp):,}"
            )
