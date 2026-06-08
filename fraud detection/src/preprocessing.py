"""
preprocessing.py
================
Encoding, scaling, and class-imbalance handling for both datasets.
All transformers are fitted on training data only.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fraud_Data columns
# ---------------------------------------------------------------------------

FRAUD_CAT_COLS  = ["source", "browser", "sex", "country"]
FRAUD_NUM_COLS  = [
    "purchase_value", "age",
    "hour_of_day", "day_of_week",
    "time_since_signup_h",
    "user_txn_count", "user_mean_purchase", "device_txn_count",
]
FRAUD_DROP_COLS = [
    "user_id", "signup_time", "purchase_time", "device_id",
    "ip_address", "ip_int", "lower_int", "upper_int",
    "time_since_signup_s",
]
FRAUD_TARGET    = "class"


# ---------------------------------------------------------------------------
# CreditCard columns
# ---------------------------------------------------------------------------

CC_NUM_COLS  = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
CC_TARGET    = "Class"


# ---------------------------------------------------------------------------
# Encoding + Scaling
# ---------------------------------------------------------------------------

def encode_and_scale_fraud(
    df: pd.DataFrame,
    scaler_type: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler, list]:
    """
    One-hot encode categoricals, scale numerics for Fraud_Data.

    Returns
    -------
    df_ready   : processed DataFrame (features + target)
    scaler     : fitted scaler (for later use on test split)
    feature_cols: list of feature column names
    """
    df = df.copy()

    # Drop non-feature columns that still exist
    drop_existing = [c for c in FRAUD_DROP_COLS if c in df.columns]
    df.drop(columns=drop_existing, inplace=True)

    # One-hot encode
    cat_existing = [c for c in FRAUD_CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_existing, drop_first=False, dtype=int)

    # Scale numerics
    num_existing = [c for c in FRAUD_NUM_COLS if c in df.columns]
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    df[num_existing] = scaler.fit_transform(df[num_existing])

    feature_cols = [c for c in df.columns if c != FRAUD_TARGET]
    logger.info("Fraud encoding done. Features: %d", len(feature_cols))
    return df, scaler, feature_cols


def scale_creditcard(
    df: pd.DataFrame,
    scaler_type: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale 'Time' and 'Amount' for creditcard (V1-V28 already PCA-scaled).
    """
    df = df.copy()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in df.columns]
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    logger.info("CreditCard scaling done.")
    return df, scaler


# ---------------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        "Split: train=%d (fraud=%d), test=%d (fraud=%d)",
        len(X_train), y_train.sum(), len(X_test), y_test.sum(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Class imbalance handling  (apply on TRAINING SET ONLY)
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """
    Apply SMOTE to the training set.
    sampling_strategy=0.3 → minority becomes 30% of majority count.
    Chosen over pure undersampling to preserve majority-class information.
    """
    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5,
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE — train size: %d  |  fraud: %d (%.1f%%)",
        len(X_res), y_res.sum(), y_res.mean() * 100,
    )
    return X_res, y_res


def apply_combined_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    over_strategy: float = 0.3,
    under_strategy: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """
    Combined SMOTE + RandomUnderSampler pipeline.
    Often gives better generalisation than SMOTE alone on very large datasets.
    """
    pipeline = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=over_strategy, random_state=random_state)),
        ("under", RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)),
    ])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    logger.info(
        "After combined resampling — train: %d  |  fraud: %d (%.1f%%)",
        len(X_res), y_res.sum(), y_res.mean() * 100,
    )
    return X_res, y_res
