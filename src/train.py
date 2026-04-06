"""
train.py
--------
Training entrypoint for the equity return prediction pipeline.

Loads a Numerai-style CSV/Parquet dataset, runs feature engineering,
trains XGBoost, LightGBM, and a StackedEnsemble with era-aware
cross-validation, prints per-era correlation scores, and saves all
models to the output directory.

Usage
-----
    python src/train.py \\
        --data_path data/train.parquet \\
        --output_dir models/ \\
        --n_folds 5
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import engineer_features, select_features, ERA_COL, TARGET_COL
from src.models import XGBoostModel, LightGBMModel, StackedEnsemble, evaluate_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train equity return prediction models on a Numerai-style dataset."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training CSV or Parquet file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="Directory where trained models will be saved. (default: models/)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds. (default: 5)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=50,
        help="Number of features to select after engineering. (default: 50)",
    )
    parser.add_argument(
        "--skip_interactions",
        action="store_true",
        help="Skip interaction term generation (faster, lower memory).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COL,
        help=f"Target column name. (default: {TARGET_COL})",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    path : str
        Path to a .csv, .parquet, or .pq file.

    Returns
    -------
    pd.DataFrame
    """
    path = str(path)
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {len(df.columns):,} columns from {path}")
    return df


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def print_era_correlations(predictions: np.ndarray, targets: np.ndarray, eras: np.ndarray) -> None:
    """Print a summary table of per-era Spearman correlation scores."""
    from scipy.stats import spearmanr

    era_series = pd.Series(eras)
    unique_eras = era_series.unique()
    records = []
    for era in sorted(unique_eras):
        mask = eras == era
        if mask.sum() < 2:
            continue
        corr = float(spearmanr(predictions[mask], targets[mask])[0])
        records.append({"era": era, "spearman_corr": corr})

    era_df = pd.DataFrame(records)
    print("\n--- Per-Era Spearman Correlation ---")
    print(f"  Mean:   {era_df['spearman_corr'].mean():.4f}")
    print(f"  Std:    {era_df['spearman_corr'].std():.4f}")
    print(f"  Min:    {era_df['spearman_corr'].min():.4f}")
    print(f"  Max:    {era_df['spearman_corr'].max():.4f}")
    print(f"  Positive eras: {(era_df['spearman_corr'] > 0).sum()} / {len(era_df)}")
    print()
    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_rows", 20):
        print(era_df.to_string(index=False))
    print()


def print_metrics(metrics: dict, label: str) -> None:
    """Print evaluation metrics in a formatted block."""
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Pearson correlation : {metrics['pearson_corr']:.4f}")
    print(f"  Spearman correlation: {metrics['spearman_corr']:.4f}")
    print(f"  Sharpe ratio        : {metrics['sharpe_ratio']:.4f}")
    print(f"  Max drawdown        : {metrics['max_drawdown']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ---- Setup output dir ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("\n[1/5] Loading data...")
    df = load_data(args.data_path)

    # ---- Validate required columns ----
    if args.target not in df.columns:
        print(f"ERROR: Target column '{args.target}' not found in dataset.")
        print(f"Available columns: {list(df.columns[:20])} ...")
        sys.exit(1)

    has_era = ERA_COL in df.columns
    if not has_era:
        print(f"WARNING: '{ERA_COL}' column not found. Era-grouped CV will be disabled.")

    # ---- Feature engineering ----
    print("\n[2/5] Engineering features...")
    t0 = time.time()
    df_engineered, all_feature_cols = engineer_features(
        df,
        add_interactions=not args.skip_interactions,
    )
    print(f"  Engineered {len(all_feature_cols):,} features in {time.time() - t0:.1f}s")

    # ---- Feature selection ----
    print(f"\n[3/5] Selecting top {args.n_features} features...")
    selected_features = select_features(
        df_engineered,
        target=args.target,
        n=args.n_features,
    )
    print(f"  Selected {len(selected_features)} features.")

    X = df_engineered[selected_features].fillna(0.0).values
    y = df_engineered[args.target].values
    eras = df_engineered[ERA_COL].values if has_era else None

    # ---- Train base models ----
    print(f"\n[4/5] Training base models with {args.n_folds}-fold CV...")

    xgb_model = XGBoostModel()
    lgb_model = LightGBMModel()

    print("\n  -- XGBoost --")
    xgb_cv = xgb_model.cross_validate(X, y, eras=eras, n_folds=args.n_folds)
    print(f"  CV Spearman: {xgb_cv['mean_spearman']:.4f} ± {np.std(xgb_cv['val_spearman']):.4f}")
    xgb_model.fit(X, y)
    xgb_model.save(str(output_dir / "xgboost_model.joblib"))
    print(f"  Saved → {output_dir / 'xgboost_model.joblib'}")

    print("\n  -- LightGBM --")
    lgb_cv = lgb_model.cross_validate(X, y, eras=eras, n_folds=args.n_folds)
    print(f"  CV Spearman: {lgb_cv['mean_spearman']:.4f} ± {np.std(lgb_cv['val_spearman']):.4f}")
    lgb_model.fit(X, y)
    lgb_model.save(str(output_dir / "lightgbm_model.joblib"))
    print(f"  Saved → {output_dir / 'lightgbm_model.joblib'}")

    # ---- Train stacked ensemble ----
    print("\n  -- Stacked Ensemble --")
    ensemble = StackedEnsemble(
        base_models=[XGBoostModel(), LightGBMModel()],
        meta_alpha=1.0,
        n_folds=args.n_folds,
    )
    ensemble.fit(X, y, eras=eras)
    ensemble.save(str(output_dir / "ensemble.joblib"))
    print(f"  Saved → {output_dir / 'ensemble.joblib'}")

    # ---- Evaluate on training data (in-sample, for logging) ----
    print("\n[5/5] Evaluating ensemble on training data (OOF)...")
    oof_preds = (
        xgb_cv["oof_predictions"] + lgb_cv["oof_predictions"]
    ) / 2.0
    metrics = evaluate_model(oof_preds, y, eras=eras)
    print_metrics(metrics, "Ensemble OOF Evaluation")

    if eras is not None:
        print_era_correlations(oof_preds, y, eras)

    # ---- Save selected features list ----
    feature_list_path = output_dir / "selected_features.txt"
    with open(feature_list_path, "w") as f:
        f.write("\n".join(selected_features))
    print(f"  Selected features saved → {feature_list_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
