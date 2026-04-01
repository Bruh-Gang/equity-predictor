"""
predict.py
----------
Inference entrypoint for the equity return prediction pipeline.

Loads a saved StackedEnsemble (or any model saved with joblib), runs the
same feature engineering pipeline on new data, and writes a CSV of
prediction scores suitable for submission to the Numerai tournament.

Usage
-----
    python src/predict.py \\
        --model_path models/ensemble.pkl \\
        --data_path data/live.parquet \\
        --output_path predictions/live_predictions.csv

Output CSV columns:
    id              — stock identifier (if present in input data)
    prediction      — scaled prediction score in [0, 1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import engineer_features, ERA_COL, TARGET_COL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate equity return predictions using a saved ensemble model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model file (.pkl).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input CSV or Parquet file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions/predictions.csv",
        help="Path for the output CSV. (default: predictions/predictions.csv)",
    )
    parser.add_argument(
        "--feature_list",
        type=str,
        default=None,
        help="Optional path to a text file listing selected feature names (one per line). "
             "If not provided, all engineered features are used.",
    )
    parser.add_argument(
        "--skip_interactions",
        action="store_true",
        help="Skip interaction term generation (must match training configuration).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV or Parquet file.

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame
    """
    path = str(path)
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def scale_predictions(preds: np.ndarray) -> np.ndarray:
    """
    Scale raw model outputs to a [0, 1] range via rank-based normalization.

    Numerai expects predictions in [0, 1]. This function applies a
    percentile-based rank transform, which also makes the submission
    distribution uniform across the prediction range.

    Parameters
    ----------
    preds : np.ndarray

    Returns
    -------
    np.ndarray
        Predictions in [0, 1].
    """
    ranks = pd.Series(preds).rank(pct=True).values
    return ranks


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # ---- Load model ----
    print(f"\n[1/4] Loading model from {args.model_path}...")
    if not Path(args.model_path).exists():
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)
    model = joblib.load(args.model_path)
    print(f"  Model type: {type(model).__name__}")

    # ---- Load feature list ----
    selected_features = None
    if args.feature_list:
        feature_list_path = Path(args.feature_list)
        if feature_list_path.exists():
            with open(feature_list_path) as f:
                selected_features = [line.strip() for line in f if line.strip()]
            print(f"  Loaded {len(selected_features)} feature names from {args.feature_list}")
        else:
            print(f"  WARNING: Feature list file not found at {args.feature_list}. Using all features.")

    # ---- Load data ----
    print(f"\n[2/4] Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    print(f"  {len(df):,} rows × {len(df.columns):,} columns")

    has_id = "id" in df.columns
    ids = df["id"].values if has_id else np.arange(len(df))

    # Remove target if present (live data should not have target)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # ---- Feature engineering ----
    print("\n[3/4] Engineering features...")
    df_engineered, all_feature_cols = engineer_features(
        df,
        add_interactions=not args.skip_interactions,
    )
    print(f"  {len(all_feature_cols):,} features after engineering")

    # ---- Build feature matrix ----
    if selected_features is not None:
        missing = [f for f in selected_features if f not in df_engineered.columns]
        if missing:
            print(f"  WARNING: {len(missing)} selected features not found in engineered data.")
            selected_features = [f for f in selected_features if f in df_engineered.columns]
        X = df_engineered[selected_features].fillna(0.0).values
        print(f"  Using {len(selected_features)} pre-selected features.")
    else:
        X = df_engineered[all_feature_cols].fillna(0.0).values
        print(f"  Using all {len(all_feature_cols)} engineered features.")

    # ---- Generate predictions ----
    print("\n[4/4] Generating predictions...")
    raw_preds = model.predict(X)
    scaled_preds = scale_predictions(raw_preds)

    # ---- Write output ----
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = pd.DataFrame({
        "id": ids,
        "prediction": scaled_preds,
    })
    output_df.to_csv(str(output_path), index=False)

    print(f"\n  Predictions written to {output_path}")
    print(f"  Rows: {len(output_df):,}")
    print(f"  Prediction range: [{scaled_preds.min():.4f}, {scaled_preds.max():.4f}]")
    print(f"  Prediction mean:  {scaled_preds.mean():.4f}")
    print(f"  Prediction std:   {scaled_preds.std():.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
