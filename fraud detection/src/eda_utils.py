"""
eda_utils.py
============
Reusable EDA plotting helpers used by both EDA notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Consistent colour palette
PALETTE = {"legit": "#2196F3", "fraud": "#E53935"}


def plot_class_imbalance(y: pd.Series, title: str = "Class Distribution") -> None:
    """Bar chart + pie chart of class distribution side-by-side."""
    counts = y.value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    colors = [PALETTE["legit"], PALETTE["fraud"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontsize=11)
    axes[0].set_title(f"{title}\n(Bar Chart)", fontsize=13)
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, counts.max() * 1.12)

    # Pie
    axes[1].pie(
        counts.values, labels=labels, colors=colors, autopct="%1.2f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title(f"{title}\n(Proportion)", fontsize=13)

    plt.suptitle(
        f"Total: {len(y):,}  |  Fraud rate: {y.mean()*100:.2f}%",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame, cols: list, target: str) -> None:
    """
    For each numeric column, plot overlapping KDE for fraud vs. legitimate.
    """
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        for cls, label, color in [(0, "Legit", PALETTE["legit"]), (1, "Fraud", PALETTE["fraud"])]:
            subset = df[df[target] == cls][col].dropna()
            subset.plot.kde(ax=axes[i], label=label, color=color, linewidth=2)
        axes[i].set_title(col, fontsize=11)
        axes[i].legend()
        axes[i].set_xlabel("")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions: Fraud vs. Legitimate", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_categorical_fraud_rate(df: pd.DataFrame, cols: list, target: str) -> None:
    """
    For each categorical column, bar chart of fraud rate per category.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        rates = (
            df.groupby(col)[target].mean().sort_values(ascending=False) * 100
        )
        rates.plot(kind="bar", ax=ax, color=PALETTE["fraud"], alpha=0.85, edgecolor="white")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(f"Fraud Rate by {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Fraud Rate (%)")
        ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """Heatmap of numeric correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(min(len(corr), 18), min(len(corr), 14)))
    sns.heatmap(
        corr, mask=mask, annot=len(corr) <= 15, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.4,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_country_fraud(df: pd.DataFrame, target: str = "class", min_txns: int = 100) -> None:
    """Top-15 countries by fraud rate (min_txns threshold to remove noise)."""
    stats = (
        df.groupby("country")[target]
        .agg(fraud_rate="mean", transactions="count")
        .query(f"transactions >= {min_txns}")
        .sort_values("fraud_rate", ascending=False)
        .head(15)
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    stats["fraud_rate"].mul(100).plot(kind="bar", ax=axes[0], color=PALETTE["fraud"], alpha=0.85)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_title("Top 15 Countries by Fraud Rate", fontsize=12)
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Fraud Rate (%)")
    axes[0].tick_params(axis="x", rotation=45)

    stats["transactions"].plot(kind="bar", ax=axes[1], color=PALETTE["legit"], alpha=0.85)
    axes[1].set_title("Transaction Volume (same countries)", fontsize=12)
    axes[1].set_xlabel("Country")
    axes[1].set_ylabel("Transaction Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def summarise_imbalance(y_before: pd.Series, y_after: pd.Series, label: str = "") -> None:
    """Print a clean before/after imbalance summary table."""
    def _stats(y):
        c = y.value_counts().sort_index()
        return c.get(0, 0), c.get(1, 0), y.mean() * 100

    l0, f0, r0 = _stats(y_before)
    l1, f1, r1 = _stats(y_after)

    print(f"\n{'='*50}")
    print(f"  Class Imbalance Summary  {label}")
    print(f"{'='*50}")
    print(f"  {'':20s} {'Before':>10s}  {'After':>10s}")
    print(f"  {'Legitimate (0)':20s} {l0:>10,}  {l1:>10,}")
    print(f"  {'Fraud (1)':20s} {f0:>10,}  {f1:>10,}")
    print(f"  {'Fraud rate':20s} {r0:>9.2f}%  {r1:>9.2f}%")
    print(f"{'='*50}\n")
