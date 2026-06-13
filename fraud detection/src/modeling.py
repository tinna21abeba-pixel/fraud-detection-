"""
modeling.py
===========
Model building, training, evaluation, and comparison for Task 2.
Handles both Fraud_Data (e-commerce) and creditcard datasets.

Usage
-----
from src.modeling import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    evaluate_model,
    cross_validate_model,
    compare_models,
    save_model,
    load_model,
)
"""

import os
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """
    Logistic Regression baseline.
    - class_weight='balanced' handles residual imbalance after resampling.
    - max_iter=1000 avoids convergence warnings on high-dimensional data.
    """
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info("Logistic Regression model built.")
    return model


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int = None,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Random Forest ensemble model.
    - n_estimators=300: enough trees for stable predictions.
    - class_weight='balanced_subsample': handles imbalance per tree.
    - n_jobs=-1: parallel training.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info("Random Forest model built (n_estimators=%d).", n_estimators)
    return model


def build_xgboost(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: float = None,
    random_state: int = 42,
) -> XGBClassifier:
    """
    XGBoost ensemble model.
    - scale_pos_weight: set to (neg/pos) ratio for imbalanced data when NOT using SMOTE.
    - subsample=0.8, colsample_bytree=0.8: regularization to prevent overfitting.
    - eval_metric='aucpr': AUC-PR better suited than AUC-ROC for imbalanced data.
    """
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(**params)
    logger.info("XGBoost model built (n_estimators=%d, max_depth=%d).", n_estimators, max_depth)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    plot: bool = True,
    save_dir: str = None,
) -> dict:
    """
    Comprehensive evaluation: AUC-PR, F1 (threshold=0.5), ROC-AUC,
    confusion matrix, and precision-recall curve.

    Returns a dict of scalar metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Scalar metrics
    aucpr   = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, digits=4)

    metrics = {
        "model":   model_name,
        "AUC-PR":  round(aucpr,   4),
        "ROC-AUC": round(roc_auc, 4),
        "F1":      round(f1,      4),
    }

    logger.info(
        "[%s] AUC-PR=%.4f | ROC-AUC=%.4f | F1=%.4f",
        model_name, aucpr, roc_auc, f1,
    )
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  AUC-PR  : {aucpr:.4f}  (primary metric)")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{model_name} — Evaluation", fontsize=14, fontweight="bold")

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Confusion Matrix")

        # 2. Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(recall, precision, color="darkorange", lw=2,
                     label=f"AUC-PR = {aucpr:.4f}")
        axes[1].axhline(y=y_test.mean(), color="navy", linestyle="--",
                        label=f"Baseline (fraud rate={y_test.mean():.3f})")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].legend(loc="upper right")
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])

        # 3. ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[2].plot(fpr, tpr, color="steelblue", lw=2,
                     label=f"ROC-AUC = {roc_auc:.4f}")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].set_title("ROC Curve")
        axes[2].legend(loc="lower right")

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_evaluation.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            logger.info("Saved evaluation plot: %s", fname)
        plt.show()

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Stratified K-Fold cross-validation.
    Reports mean ± std for AUC-PR, ROC-AUC, and F1.

    NOTE: Cross-validate on SMOTE-resampled data if you want resampling per fold,
    or pass original training data to get unbiased per-fold resampling.
    For simplicity here we CV on whatever X_train/y_train is passed in.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "aucpr":   "average_precision",
        "roc_auc": "roc_auc",
        "f1":      "f1",
    }

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {
        "model":          model_name,
        "CV AUC-PR":      f"{cv_results['test_aucpr'].mean():.4f} ± {cv_results['test_aucpr'].std():.4f}",
        "CV ROC-AUC":     f"{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}",
        "CV F1":          f"{cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}",
    }

    print(f"\nCross-Validation ({n_splits}-fold) — {model_name}")
    print(f"  AUC-PR  : {summary['CV AUC-PR']}")
    print(f"  ROC-AUC : {summary['CV ROC-AUC']}")
    print(f"  F1      : {summary['CV F1']}")

    logger.info("[%s] CV done: %s", model_name, summary)
    return summary


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_models(metrics_list: list, title: str = "Model Comparison") -> pd.DataFrame:
    """
    Display a side-by-side comparison table and bar chart from a list of metrics dicts.
    Each dict should have keys: model, AUC-PR, ROC-AUC, F1.

    Example
    -------
    results = [
        evaluate_model(lr,  X_test, y_test, "Logistic Regression", plot=False),
        evaluate_model(rf,  X_test, y_test, "Random Forest",        plot=False),
        evaluate_model(xgb, X_test, y_test, "XGBoost",              plot=False),
    ]
    compare_models(results)
    """
    df = pd.DataFrame(metrics_list).set_index("model")

    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(df.to_string())

    # Bar chart
    ax = df.plot(kind="bar", figsize=(10, 5), rot=0, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.show()

    return df


# ---------------------------------------------------------------------------
# Feature importance plot (for tree-based models)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str = "Model",
    top_n: int = 20,
    save_dir: str = None,
) -> None:
    """Plot built-in feature importances for tree-based models (RF / XGBoost)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model has no feature_importances_ attribute.")
        return

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top = fi.head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    top.sort_values().plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_feature_importance.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance plot: %s", fname)
    plt.show()


# ---------------------------------------------------------------------------
# Persist models
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    """Serialize model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved: %s", path)


def load_model(path: str):
    """Load a serialized model from disk."""
    model = joblib.load(path)
    logger.info("Model loaded: %s", path)
    return model
