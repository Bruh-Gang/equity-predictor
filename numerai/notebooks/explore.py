"""
Numerai Data Exploration & Model Diagnostics
---------------------------------------------
Run this interactively in Jupyter or as a standalone script.
Useful for understanding your data, checking model health, and debugging.

Usage in Jupyter:
    %run notebooks/explore.py

Usage as script:
    python notebooks/explore.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    per_era_corr,
    mean_std_sharpe,
    max_drawdown,
    smart_sharpe,
    validation_metrics,
    feature_exposure,
    max_feature_exposure,
)


def plot_era_corr(era_scores: pd.Series, title: str = "Per-Era Correlation", save_path: str = None):
    """Plot per-era correlation with trend line."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Per-era bar chart
    ax = axes[0]
    colors = ["green" if v > 0 else "red" for v in era_scores]
    ax.bar(range(len(era_scores)), era_scores, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(era_scores.mean(), color="blue", linestyle="--", linewidth=1.5,
               label=f"Mean: {era_scores.mean():.4f}")
    ax.set_title(f"{title} | Sharpe={era_scores.mean()/era_scores.std()*np.sqrt(52):.2f}")
    ax.set_xlabel("Era")
    ax.set_ylabel("Correlation")
    ax.legend()

    # Cumulative sum
    ax = axes[1]
    cumsum = era_scores.cumsum()
    ax.plot(cumsum.values, color="steelblue", linewidth=1.5)
    ax.fill_between(range(len(cumsum)), cumsum.values, alpha=0.15, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Cumulative Correlation | Max DD={max_drawdown(era_scores):.4f}")
    ax.set_xlabel("Era")
    ax.set_ylabel("Cumulative Corr")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.close()


def plot_feature_exposure(
    df: pd.DataFrame,
    pred_col: str,
    feature_cols: list,
    top_n: int = 20,
    era_col: str = "era",
    save_path: str = None,
):
    """Plot max feature exposure."""
    max_exp = max_feature_exposure(df, pred_col, feature_cols, era_col)
    top = max_exp.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    top.plot(kind="bar", ax=ax, color="coral", alpha=0.8)
    ax.axhline(0.1, color="red", linestyle="--", label="Danger threshold (0.1)")
    ax.set_title(f"Top {top_n} Max Feature Exposures")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Max Absolute Exposure")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.close()


def compare_models(
    val_df: pd.DataFrame,
    model_preds: dict,  # {model_name: predictions_array}
    target_col: str = "target",
    era_col: str = "era",
) -> pd.DataFrame:
    """Compare multiple models on validation metrics."""
    rows = []
    for name, preds in model_preds.items():
        df = val_df.copy()
        df["_pred"] = preds
        metrics = validation_metrics(df, "_pred", target_col, era_col)
        metrics["model"] = name
        rows.append(metrics)
    comparison = pd.DataFrame(rows).set_index("model")
    print("\n=== Model Comparison ===")
    print(comparison.to_string(float_format="{:.5f}".format))
    return comparison


def correlation_matrix(model_preds: dict) -> pd.DataFrame:
    """Compute pairwise correlation between model predictions (ideally should be low!)."""
    pred_df = pd.DataFrame(model_preds)
    corr = pred_df.corr()
    print("\n=== Model Prediction Correlation Matrix ===")
    print("(Lower = more diverse ensemble = better MMC)")
    print(corr.to_string(float_format="{:.3f}".format))
    return corr


if __name__ == "__main__":
    print("Numerai Diagnostic Tools loaded.")
    print("Import and use functions like:")
    print("  plot_era_corr(era_scores)")
    print("  compare_models(val_df, {'lgbm': preds1, 'xgb': preds2})")
    print("  correlation_matrix({'lgbm': preds1, 'xgb': preds2})")
