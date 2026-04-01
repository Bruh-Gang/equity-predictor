"""
feature_analysis.py
-------------------
Feature importance and stability analysis for trained equity return models.

Generates three plot types:
  1. Feature importance bar charts (XGBoost gain, LightGBM split/gain)
  2. Spearman correlation of features with target, plotted per era
  3. Feature stability analysis: std of importance across CV folds

Usage
-----
    python analysis/feature_analysis.py \\
        --model_path models/ \\
        --data_path data/train.parquet \\
        --output_dir analysis/plots/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import engineer_features, select_features, ERA_COL, TARGET_COL
from src.models import XGBoostModel, LightGBMModel, _era_kfold_indices

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
PALETTE = "Blues_r"
FIG_DPI = 150


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature importance and stability analysis.")
    parser.add_argument("--model_path", type=str, default="models/",
                        help="Directory containing saved .pkl model files.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training CSV or Parquet file.")
    parser.add_argument("--output_dir", type=str, default="analysis/plots/",
                        help="Directory to save output plots.")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of top features to display in importance plots.")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Folds for stability analysis.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plot 1: Feature importance bar chart
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    title: str,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """
    Plot a horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importances : np.ndarray
        Array of importance scores, one per feature.
    feature_names : list of str
    title : str
        Plot title (e.g., "XGBoost Feature Importance (Gain)").
    output_path : Path
        File path to save the figure.
    top_n : int
        Number of features to display.
    """
    n = min(top_n, len(importances))
    idx = np.argsort(importances)[::-1][:n]
    top_imp = importances[idx]
    top_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.35)))
    colors = sns.color_palette(PALETTE, n_colors=n)
    bars = ax.barh(range(n), top_imp[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(
                width * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}", va="center", fontsize=7.5, color="#333333",
            )

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Feature-target correlation across eras
# ---------------------------------------------------------------------------


def plot_era_feature_correlations(
    df: pd.DataFrame,
    feature_cols: list,
    top_n: int = 20,
    output_path: Path = None,
) -> None:
    """
    Compute and plot per-era Spearman correlation of the top-N features
    with the target. Displayed as a heatmap (features × eras).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ERA_COL and TARGET_COL.
    feature_cols : list of str
    top_n : int
        Number of top features (by mean absolute era-correlation) to plot.
    output_path : Path
    """
    if ERA_COL not in df.columns or TARGET_COL not in df.columns:
        print("  Skipping era correlation plot (missing era or target column).")
        return

    unique_eras = sorted(df[ERA_COL].unique())
    corr_matrix = {}

    for feat in feature_cols[:50]:  # cap to 50 candidates for speed
        era_corrs = []
        for era in unique_eras:
            mask = df[ERA_COL] == era
            sub = df[mask]
            if len(sub) < 5:
                era_corrs.append(np.nan)
                continue
            corr = float(spearmanr(sub[feat].fillna(0), sub[TARGET_COL].fillna(0))[0])
            era_corrs.append(corr)
        corr_matrix[feat] = era_corrs

    corr_df = pd.DataFrame(corr_matrix, index=unique_eras).T  # shape: (features, eras)
    mean_abs = corr_df.abs().mean(axis=1)
    top_features = mean_abs.nlargest(top_n).index.tolist()
    corr_df_top = corr_df.loc[top_features]

    # Truncate era labels for readability
    era_labels = [str(e)[:6] for e in unique_eras]
    fig, ax = plt.subplots(figsize=(max(12, len(unique_eras) * 0.4), max(8, top_n * 0.4)))
    sns.heatmap(
        corr_df_top,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.15,
        vmax=0.15,
        xticklabels=era_labels,
        yticklabels=top_features,
        linewidths=0.3,
        cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
    )
    ax.set_title(f"Feature–Target Spearman Correlation by Era (top {top_n})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Era", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=90)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    if output_path:
        fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Feature stability (std of importance across CV folds)
# ---------------------------------------------------------------------------


def plot_feature_stability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    eras: np.ndarray,
    model_class,
    title: str,
    output_path: Path,
    n_folds: int = 5,
    top_n: int = 30,
) -> None:
    """
    Train the model on each CV fold separately and plot the standard deviation
    of feature importances across folds as a proxy for stability.

    Stable features have low std; unstable features vary significantly
    between folds and may be noisy signals.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    feature_names : list of str
    eras : np.ndarray or None
    model_class : type
        XGBoostModel or LightGBMModel.
    title : str
    output_path : Path
    n_folds : int
    top_n : int
    """
    if eras is not None:
        fold_indices = _era_kfold_indices(eras, n_folds)
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_indices = list(kf.split(X))

    fold_importances = []
    for train_idx, _ in fold_indices:
        m = model_class()
        m.fit(X[train_idx], y[train_idx])
        if m.feature_importances_ is not None:
            fold_importances.append(m.feature_importances_)

    if not fold_importances:
        print(f"  Skipping stability plot for {title} — no importances available.")
        return

    imp_matrix = np.array(fold_importances)  # shape: (n_folds, n_features)
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0)

    # Select top-N by mean importance
    idx = np.argsort(mean_imp)[::-1][:top_n]
    names_top = [feature_names[i] for i in idx]
    mean_top = mean_imp[idx]
    std_top = std_imp[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, mean_top[::-1], xerr=std_top[::-1],
            color=sns.color_palette(PALETTE, n_colors=top_n)[::-1],
            edgecolor="white", linewidth=0.5, error_kw={"ecolor": "#666666", "capsize": 3})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_top[::-1], fontsize=9)
    ax.set_xlabel("Mean Importance ± Std (across folds)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load and engineer features ----
    print(f"\n[1/4] Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    print(f"  {len(df):,} rows loaded.")

    print("\n[2/4] Engineering features...")
    df_eng, all_feature_cols = engineer_features(df, add_interactions=False)
    feature_names = all_feature_cols
    has_era = ERA_COL in df_eng.columns
    has_target = TARGET_COL in df_eng.columns

    if not has_target:
        print("ERROR: Target column not found. Cannot perform feature analysis.")
        sys.exit(1)

    X = df_eng[feature_names].fillna(0.0).values
    y = df_eng[TARGET_COL].values
    eras = df_eng[ERA_COL].values if has_era else None

    # ---- Load or train models for importances ----
    print("\n[3/4] Fitting models for importance extraction...")
    xgb_model = XGBoostModel()
    lgb_model = LightGBMModel()
    xgb_model.fit(X, y)
    lgb_model.fit(X, y)

    # ---- Plot 1a: XGBoost importance ----
    print("\n[4/4] Generating plots...")
    if xgb_model.feature_importances_ is not None:
        plot_feature_importance(
            xgb_model.feature_importances_,
            feature_names,
            title=f"XGBoost Feature Importance (Gain) — Top {args.top_n}",
            output_path=output_dir / "xgboost_importance.png",
            top_n=args.top_n,
        )

    # ---- Plot 1b: LightGBM importance ----
    if lgb_model.feature_importances_ is not None:
        plot_feature_importance(
            lgb_model.feature_importances_,
            feature_names,
            title=f"LightGBM Feature Importance (Split) — Top {args.top_n}",
            output_path=output_dir / "lightgbm_importance.png",
            top_n=args.top_n,
        )

    # ---- Plot 2: Era-wise feature-target correlation ----
    if has_era:
        plot_era_feature_correlations(
            df_eng,
            feature_cols=feature_names,
            top_n=min(20, args.top_n),
            output_path=output_dir / "era_feature_correlations.png",
        )
    else:
        print("  Skipping era correlation heatmap (no era column).")

    # ---- Plot 3a: XGBoost stability ----
    plot_feature_stability(
        X, y,
        feature_names=feature_names,
        eras=eras,
        model_class=XGBoostModel,
        title=f"XGBoost Feature Stability (Mean ± Std across {args.n_folds} folds)",
        output_path=output_dir / "xgboost_stability.png",
        n_folds=args.n_folds,
        top_n=args.top_n,
    )

    # ---- Plot 3b: LightGBM stability ----
    plot_feature_stability(
        X, y,
        feature_names=feature_names,
        eras=eras,
        model_class=LightGBMModel,
        title=f"LightGBM Feature Stability (Mean ± Std across {args.n_folds} folds)",
        output_path=output_dir / "lightgbm_stability.png",
        n_folds=args.n_folds,
        top_n=args.top_n,
    )

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
