# Return Fraud Detection System

> Unsupervised customer clustering + pre-shipment return risk scoring for e-commerce retailers.  
> Cluster-augmented model achieves **12.8% higher AUC-PR** than a transaction-only baseline.

---

## The Problem

Mid-size e-commerce retailers lose significant margin to return fraud and policy abuse without realising it. A customer who returns 65% of orders looks identical to a first-time buyer at checkout — until you model their behavioral history. This project builds a two-layer ML system that first segments customers by return behavior, then uses those behavioral profiles to predict whether a new order will result in a return **before it ships**.

---

## Live Demo

> 🔗 [aanyame14-return-fraud.streamlit.app](#) — add your URL here  

---

## Key Findings

| Metric | Baseline (transaction only) | Cluster-augmented | Improvement |
|---|---|---|---|
| AUC-PR | 0.233 | 0.263 | +12.8% |
| AUC-ROC | 0.711 | 0.731 | +2.8% |
| Lift @ top decile | 2.62× | 2.90× | +10.7% |
| Best F1 | — | 0.290 | threshold = 0.62 |

**5 cluster features rank in the top 10 SHAP importances** — confirming the behavioral layer adds genuine predictive signal beyond raw transaction data.

---

## Customer Segments Discovered

| Cluster | Customers | Return Rate | Avg Order Value | Key Signal |
|---|---|---|---|---|
| Fraud / Serial Abusers | 442 | 65.5% | R$136 | Near-instant complaints, DBSCAN outliers |
| Policy Abusers | 4 | 30.0% | R$108 | Consistent returns across categories |
| Occasional Dissatisfied | 165 | 18.6% | R$125 | Moderate rate, normal timing |
| High-Value Fast Returners | 253 | 9.5% | R$454 | Expensive orders, same-day complaints |
| Loyal Low-Risk | 1,937 | 0.2% | R$110 | Almost never return |

**195 customers (7%)** were flagged as DBSCAN noise — behavioral outliers that don't fit any cluster and represent the highest-priority fraud investigation targets.

---

## How It Works

```
Raw Olist data
      │
      ▼
Feature engineering        Per-customer: return rate, timing, category affinity,
                           payment preference (2,801 customers with ≥2 orders)
      │
      ▼
PCA (4 components          86.3% variance explained
 → 86.3% variance)
      │
      ├──▶ Hierarchical clustering (Ward, k=5)   → cluster labels + profiles
      │
      └──▶ DBSCAN (eps=0.788, min_samples=14)    → noise/anomaly flags
                │
                ▼
      Centroid distance features (5 distances per customer)
                │
                ▼
      ┌─────────────────────┐    ┌──────────────────────────────┐
      │   Model A           │    │   Model B                    │
      │   Transaction only  │    │   Transaction + cluster feats│
      │   AUC-PR: 0.233     │    │   AUC-PR: 0.263              │
      └─────────────────────┘    └──────────────────────────────┘
                                          │
                                          ▼
                              Pre-shipment risk score
                              + intervention recommendation
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy |
| Unsupervised ML | scikit-learn (PCA, DBSCAN, Ward hierarchical) |
| Supervised ML | XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Storage | Parquet (pyarrow) |

---

## Project Structure

```
return-fraud-clustering/
├── data/
│   ├── raw/                         # Olist CSVs (gitignored — too large)
│   └── features/
│       ├── customer_features.parquet
│       └── customer_features_clustered.parquet
├── notebooks/
│   ├── 01_eda_plots.py
│   ├── 02_feature_engineering.py
│   ├── 03_clustering.py
│   ├── 04_supervised.py
│   └── 05_evaluation.py
├── src/
│   └── features.py                  # Shared feature engineering functions
├── app/
│   └── dashboard.py                 # Streamlit app
├── outputs/
│   ├── model_A.pkl                  # Baseline model
│   ├── model_B.pkl                  # Cluster-augmented model
│   ├── cluster_profiles.csv
│   └── eval_summary.txt
├── requirements.txt
└── README.md
```

---

## Dataset

**Olist Brazilian E-Commerce** — publicly available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

- 96,478 delivered orders
- 93,358 unique customers
- 2,801 customers with ≥ 2 orders (used for clustering)

**Return proxy definition:** Olist has no explicit return column. A return is defined as `order_status = "delivered"` AND `review_score ≤ 2`. Overall proxy return rate: **13.5% (train) / 9.6% (test)**.

---

## Reproduce the Results

```bash
# 1. Clone and set up
git clone https://github.com/yourusername/return-fraud-clustering.git
cd return-fraud-clustering
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Download Olist CSVs from Kaggle and place in data/raw/

# 3. Run notebooks in order
python notebooks/01_eda_plots.py
python notebooks/02_feature_engineering.py
python notebooks/03_clustering.py
python notebooks/04_supervised.py
python notebooks/05_evaluation.py

# 4. Launch dashboard
streamlit run app/dashboard.py
```

---

## Business Application

This system is designed for mid-size retailers who want to intervene **before** a high-risk order ships rather than after a return is filed. The dashboard's risk scorer takes an incoming order and outputs:

- **Return probability score** (0–100%)
- **Risk band** — Low / Medium / High with specific intervention recommendations
- **Cluster assignment** — which behavioral segment this customer belongs to
- **Baseline vs cluster-augmented score comparison** — showing how much behavioral history changes the prediction

Typical interventions triggered at high risk: require signature on delivery, restrict to credit card only, flag for manual dispatch review.

---

## Limitations

- Return proxy (review score ≤ 2) underestimates true returns — customers who return without reviewing are not captured
- Only customers with ≥ 2 orders are clustered — new customers receive transaction-only scores
- Dataset covers Brazilian e-commerce 2016–2018 — category and behavioral patterns may differ by market

---

## Author

Built as a portfolio project demonstrating applied ML for e-commerce fraud detection.  