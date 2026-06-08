"""
data_loader.py
==============
Reusable functions for loading and basic cleaning of both datasets.
"""

import pandas as pd
import numpy as np
import ipaddress
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fraud_Data.csv helpers
# ---------------------------------------------------------------------------

def load_fraud_data(path: str) -> pd.DataFrame:
    """Load Fraud_Data.csv with correct dtypes."""
    df = pd.read_csv(path)
    df["signup_time"]   = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    logger.info("Fraud_Data loaded: %d rows, %d cols", *df.shape)
    return df


def load_ip_country(path: str) -> pd.DataFrame:
    """Load IpAddress_to_Country.csv."""
    df = pd.read_csv(path)
    logger.info("IP-Country table loaded: %d rows", len(df))
    return df


def load_creditcard(path: str) -> pd.DataFrame:
    """Load creditcard.csv."""
    df = pd.read_csv(path)
    logger.info("CreditCard loaded: %d rows, %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate, fix types, and validate Fraud_Data.
    Returns cleaned copy.
    """
    df = df.copy()
    n_before = len(df)

    # 1. Drop exact duplicates
    df.drop_duplicates(inplace=True)
    logger.info("Dropped %d duplicate rows", n_before - len(df))

    # 2. Ensure datetimes
    for col in ["signup_time", "purchase_time"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    # 3. purchase_time must be >= signup_time (sanity check)
    bad_time = df["purchase_time"] < df["signup_time"]
    if bad_time.any():
        logger.warning("Found %d rows where purchase_time < signup_time — dropping", bad_time.sum())
        df = df[~bad_time]

    # 4. Non-negative purchase value
    df = df[df["purchase_value"] > 0]

    # 5. Age sanity (1–120)
    df = df[(df["age"] >= 1) & (df["age"] <= 120)]

    logger.info("Clean Fraud_Data: %d rows remain", len(df))
    return df.reset_index(drop=True)


def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate and validate creditcard dataset.
    """
    df = df.copy()
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("CreditCard: dropped %d duplicates", n_before - len(df))
    df = df[df["Amount"] >= 0]
    logger.info("CreditCard clean: %d rows remain", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IP → Country mapping
# ---------------------------------------------------------------------------

def ip_to_int(ip_val) -> float:
    """Convert dotted-decimal or numeric IP to integer; NaN on failure."""
    try:
        s = str(ip_val).strip()
        # If the value is already a plain float/int string (as in Fraud_Data)
        return float(s)
    except (ValueError, AttributeError):
        try:
            return float(int(ipaddress.ip_address(s)))
        except Exception:
            return np.nan


def merge_ip_country(df: pd.DataFrame, df_ip: pd.DataFrame) -> pd.DataFrame:
    """
    Range-based IP-to-country merge using merge_asof.
    Adds a 'country' column to df.
    """
    df = df.copy()
    df_ip = df_ip.copy()

    # Ensure integer columns exist
    df["ip_int"]            = df["ip_address"].apply(ip_to_int)
    df_ip["lower_int"]      = df_ip["lower_bound_ip_address"].apply(ip_to_int)
    df_ip["upper_int"]      = df_ip["upper_bound_ip_address"].apply(ip_to_int)

    df_sorted    = df.sort_values("ip_int").reset_index(drop=True)
    df_ip_sorted = df_ip.sort_values("lower_int").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sorted,
        df_ip_sorted[["lower_int", "upper_int", "country"]],
        left_on="ip_int",
        right_on="lower_int",
        direction="backward",
    )

    # Null out IPs that fall outside the matched range
    out_of_range = merged["ip_int"] > merged["upper_int"]
    merged.loc[out_of_range, "country"] = "Unknown"
    merged["country"].fillna("Unknown", inplace=True)

    mapped = (merged["country"] != "Unknown").sum()
    logger.info("IP-country merge: %d/%d rows mapped", mapped, len(merged))
    return merged.reset_index(drop=True)
