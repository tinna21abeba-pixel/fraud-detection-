"""
feature_engineering.py
=======================
All feature engineering for the Fraud_Data (e-commerce) dataset.
Credit-card dataset needs no additional feature engineering beyond scaling.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day, day_of_week, time_since_signup (seconds & hours)."""
    df = df.copy()

    df["hour_of_day"]        = df["purchase_time"].dt.hour
    df["day_of_week"]        = df["purchase_time"].dt.dayofweek   # 0=Mon

    # time_since_signup in seconds (raw) and hours (rounded to 2 dp)
    df["time_since_signup_s"]  = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds()
    df["time_since_signup_h"]  = df["time_since_signup_s"] / 3600.0

    logger.info("Time features added: hour_of_day, day_of_week, time_since_signup_h")
    return df


# ---------------------------------------------------------------------------
# Transaction velocity features
# ---------------------------------------------------------------------------

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user transaction count and mean purchase value.
    Computed over the whole dataset (training only; fit on train, transform test).
    """
    df = df.copy()

    user_stats = (
        df.groupby("user_id")["purchase_value"]
        .agg(user_txn_count="count", user_mean_purchase="mean")
        .reset_index()
    )
    df = df.merge(user_stats, on="user_id", how="left")

    # Device-level count (same device used many times → velocity signal)
    device_stats = (
        df.groupby("device_id")["purchase_value"]
        .agg(device_txn_count="count")
        .reset_index()
    )
    df = df.merge(device_stats, on="device_id", how="left")

    logger.info("Velocity features added: user_txn_count, user_mean_purchase, device_txn_count")
    return df


# ---------------------------------------------------------------------------
# All feature engineering combined
# ---------------------------------------------------------------------------

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline: apply all feature engineering steps in order.
    Expects a DataFrame that has already been cleaned and had 'country' merged in.
    """
    df = add_time_features(df)
    df = add_velocity_features(df)
    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df
