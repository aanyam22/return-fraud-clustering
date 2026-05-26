# =============================================================================
# src/features.py
# Reusable feature engineering functions.
# Called by both 02_feature_engineering.py and app/dashboard.py
# =============================================================================

import pandas as pd
import numpy as np


# Exact category columns produced by 02_feature_engineering.py
CATEGORY_COLS = [
    "returned_cama_mesa_banho",
    "returned_moveis_decoracao",
    "returned_informatica_acessorios",
    "returned_beleza_saude",
    "returned_esporte_lazer",
    "returned_utilidades_domesticas",
    "returned_relogios_presentes",
    "returned_telefonia",
]

# Exact payment dummies produced by 02_feature_engineering.py
PAYMENT_COLS = ["pay_boleto", "pay_credit_card", "pay_debit_card", "pay_voucher"]

# Model A feature columns (transaction only)
TXN_COLS = [
    "total_value",
    "payment_type_enc",
    "customer_tenure_days",
    "seller_return_rate",
    "category_return_rate",
]

# Model B feature columns (transaction + cluster)
ALL_COLS = [
    "total_value",
    "payment_type_enc",
    "customer_tenure_days",
    "seller_return_rate",
    "category_return_rate",
    "return_rate",
    "avg_order_value",
    "avg_days_to_review",
    "min_days_to_review",
    "dbscan_noise",
    "cluster_dbscan",
    "cluster_1.0",
    "cluster_2.0",
    "cluster_3.0",
    "cluster_4.0",
    "cluster_5.0",
    "dist_to_cluster_1",
    "dist_to_cluster_2",
    "dist_to_cluster_3",
    "dist_to_cluster_4",
    "dist_to_cluster_5",
]

CLUSTER_NAMES = {
    1: "Policy Abusers",
    2: "Occasional Dissatisfied",
    3: "High-Value Fast Returners",
    4: "Fraud / Serial Abusers",
    5: "Loyal Low-Risk",
}

CLUSTER_COLORS = {
    "Fraud / Serial Abusers":      "#e05c5c",
    "Policy Abusers":              "#f5a623",
    "Occasional Dissatisfied":     "#f5d623",
    "High-Value Fast Returners":   "#4f8ef7",
    "Loyal Low-Risk":              "#3ecf8e",
}

# Payment type encoding (must match LabelEncoder order from 04_supervised.py)
# LabelEncoder sorts alphabetically: boleto=0, credit_card=1, debit_card=2, voucher=3
PAYMENT_TYPE_ENC = {
    "boleto":      0,
    "credit_card": 1,
    "debit_card":  2,
    "voucher":     3,
    "unknown":     4,
}


def load_raw_tables(data_dir="data/raw"):
    """Load all Olist CSVs and return a dict of DataFrames."""
    import os
    d = data_dir.rstrip("/") + "/"
    orders    = pd.read_csv(d + "olist_orders_dataset.csv", parse_dates=[
                    "order_purchase_timestamp", "order_delivered_customer_date"])
    reviews   = pd.read_csv(d + "olist_order_reviews_dataset.csv",
                    parse_dates=["review_creation_date"])
    items     = pd.read_csv(d + "olist_order_items_dataset.csv")
    payments  = pd.read_csv(d + "olist_order_payments_dataset.csv")
    customers = pd.read_csv(d + "olist_customers_dataset.csv")
    products  = pd.read_csv(d + "olist_products_dataset.csv")
    cat_map   = pd.read_csv(d + "product_category_name_translation.csv")
    products  = products.merge(cat_map, on="product_category_name", how="left")
    return dict(orders=orders, reviews=reviews, items=items, payments=payments,
                customers=customers, products=products)


def build_master_df(tables):
    """Merge raw tables into a single order-level DataFrame."""
    orders, reviews, items, payments, customers, products = (
        tables["orders"], tables["reviews"], tables["items"],
        tables["payments"], tables["customers"], tables["products"])

    rev = (reviews.sort_values("review_score")
           .drop_duplicates("order_id", keep="first")
           [["order_id", "review_score", "review_creation_date"]])
    pay = (payments.groupby("order_id")
           .agg(total_value=("payment_value", "sum"),
                payment_type=("payment_type", "first"))
           .reset_index())

    df = (orders
          .merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                 on="customer_id", how="left")
          .merge(rev,  on="order_id", how="left")
          .merge(pay,  on="order_id", how="left"))

    df["is_return"] = (
        (df["order_status"] == "delivered") & (df["review_score"] <= 2)
    ).astype(int)

    df["days_to_review"] = (
        df["review_creation_date"] - df["order_delivered_customer_date"]
    ).dt.days
    df["days_to_review"] = df["days_to_review"].where(df["days_to_review"] >= 0).clip(upper=90)

    return df, items, products


def build_transaction_features(df, items, products):
    """
    Build order-level transaction features: tenure, seller rate, category rate,
    payment type encoding. Returns df with new columns added.
    """
    # Customer tenure
    first_order = (df.groupby("customer_unique_id")["order_purchase_timestamp"]
                   .min().rename("first_order_date").reset_index())
    df = df.merge(first_order, on="customer_unique_id", how="left")
    df["customer_tenure_days"] = (
        df["order_purchase_timestamp"] - df["first_order_date"]
    ).dt.days.fillna(0)

    # Seller historical return rate
    order_seller = (items[["order_id", "seller_id"]].drop_duplicates("order_id")
                    .merge(df[["order_id", "is_return"]], on="order_id", how="left"))
    seller_rate = (order_seller.groupby("seller_id")["is_return"]
                   .mean().rename("seller_return_rate").reset_index())
    df = df.merge(items[["order_id", "seller_id"]].drop_duplicates("order_id"),
                  on="order_id", how="left")
    df = df.merge(seller_rate, on="seller_id", how="left")

    # Category historical return rate
    order_cat = (items[["order_id", "product_id"]].drop_duplicates("order_id")
                 .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
                 .merge(df[["order_id", "is_return"]], on="order_id", how="left"))
    cat_rate = (order_cat.groupby("product_category_name")["is_return"]
                .mean().rename("category_return_rate").reset_index())
    df = df.merge(items[["order_id", "product_id"]].drop_duplicates("order_id"),
                  on="order_id", how="left")
    df = df.merge(products[["product_id", "product_category_name"]].drop_duplicates("product_id"),
                  on="product_id", how="left")
    df = df.merge(cat_rate, on="product_category_name", how="left")

    # Payment type encoding (alphabetical label encoding)
    df["payment_type_enc"] = df["payment_type"].fillna("unknown").map(PAYMENT_TYPE_ENC).fillna(4)

    return df


def encode_single_order(order_value, payment_type, tenure_days,
                         seller_return_rate, category_return_rate,
                         customer_return_rate, avg_order_value,
                         avg_days_to_review, min_days_to_review,
                         cluster_hier, dbscan_noise, cluster_dbscan,
                         dist_to_clusters):
    """
    Build a single-row feature dict for the Streamlit risk scorer.
    dist_to_clusters: list of 5 floats [d1, d2, d3, d4, d5]
    cluster_hier: int 1-5
    """
    row = {
        "total_value":           order_value,
        "payment_type_enc":      PAYMENT_TYPE_ENC.get(payment_type, 4),
        "customer_tenure_days":  tenure_days,
        "seller_return_rate":    seller_return_rate,
        "category_return_rate":  category_return_rate,
        "return_rate":           customer_return_rate,
        "avg_order_value":       avg_order_value,
        "avg_days_to_review":    avg_days_to_review,
        "min_days_to_review":    min_days_to_review,
        "dbscan_noise":          dbscan_noise,
        "cluster_dbscan":        cluster_dbscan,
        "cluster_1.0":           1 if cluster_hier == 1 else 0,
        "cluster_2.0":           1 if cluster_hier == 2 else 0,
        "cluster_3.0":           1 if cluster_hier == 3 else 0,
        "cluster_4.0":           1 if cluster_hier == 4 else 0,
        "cluster_5.0":           1 if cluster_hier == 5 else 0,
        "dist_to_cluster_1":     dist_to_clusters[0],
        "dist_to_cluster_2":     dist_to_clusters[1],
        "dist_to_cluster_3":     dist_to_clusters[2],
        "dist_to_cluster_4":     dist_to_clusters[3],
        "dist_to_cluster_5":     dist_to_clusters[4],
    }
    return pd.DataFrame([row])[ALL_COLS].fillna(0)
