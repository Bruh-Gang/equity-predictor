"""
features.py
-----------
Feature engineering pipeline for Numerai-style equity return prediction.

Provides rolling window statistics, cross-sectional normalization,
era-neutral transformations, interaction terms, and feature selection
utilities aligned with the era-structured Numerai dataset format.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.feature_selection import mutual_info_regression
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOWS: List[int] = [5, 10, 20]
ERA_COL: str = "era"
TARGET_COL: str = "target"


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean with min_periods=1."""
    return series.rolling(window=window, min_periods=1).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling standard deviation with min_periods=2."""
    return series.rolling(window=window, min_periods=2).std().fillna(0.0)


def _rolling_skew(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling skewness with min_periods=3."""
    return series.rolling(window=window, min_periods=3).skew().fillna(0.0)


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: List[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Append rolling mean, std, and skew features for each feature column
    over multiple look-back windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by time / era order.
    feature_cols : list of str
        Columns to compute rolling statistics for.
    windows : list of int
        Look-back window sizes. Default: [5, 10, 20].

    Returns
    -------
    pd.DataFrame
        Original DataFrame extended with rolling feature columns.
        New column names follow the pattern ``{col}_roll{w}_{stat}``.
    """
    new_cols: dict = {}
    for col in feature_cols:
        series = df[col]
        for w in windows:
            new_cols[f"{col}_roll{w}_mean"] = _rolling_mean(series, w).values
            new_cols[f"{col}_roll{w}_std"] = _rolling_std(series, w).values
            new_cols[f"{col}_roll{w}_skew"] = _rolling_skew(series, w).values
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Cross-sectional rank normalization
# ---------------------------------------------------------------------------


def rank_normalize_era(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply cross-sectional rank normalization within each era.

    Each feature is ranked within its era and mapped to the [0, 1] interval
    (Gaussian rank optional). This removes distributional shift across eras
    and is the canonical pre-processing step for Numerai features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``era`` column.
    feature_cols : list of str
        Columns to normalize.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns replaced by their era-wise ranks.
    """
    if ERA_COL not in df.columns:
        # If no era column, rank globally
        df[feature_cols] = df[feature_cols].rank(pct=True)
        return df

    result = df.copy()
    for era, group in df.groupby(ERA_COL):
        idx = group.index
        result.loc[idx, feature_cols] = group[feature_cols].rank(pct=True)
    return result


# ---------------------------------------------------------------------------
# Era-neutral features
# ---------------------------------------------------------------------------


def era_neutralize(
    df: pd.DataFrame,
    feature_cols: List[str],
    proportion: float = 1.0,
) -> pd.DataFrame:
    """
    Project features orthogonal to era means to isolate stock-specific signal.

    For each era, the era-mean is subtracted (scaled by ``proportion``),
    eliminating common era-level trends from individual feature values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``era`` column.
    feature_cols : list of str
        Columns to neutralize.
    proportion : float
        Fraction of era mean to subtract. 1.0 = fully neutralized.

    Returns
    -------
    pd.DataFrame
        DataFrame with era-neutralized feature columns (in-place replacement).
    """
    if ERA_COL not in df.columns:
        return df

    result = df.copy()
    for era, group in df.groupby(ERA_COL):
        idx = group.index
        era_means = group[feature_cols].mean()
        result.loc[idx, feature_cols] = (
            group[feature_cols] - proportion * era_means
        )
    return result


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------


def add_interaction_terms(
    df: pd.DataFrame,
    feature_cols: List[str],
    top_n: int = 10,
    target_col: Optional[str] = TARGET_COL,
) -> pd.DataFrame:
    """
    Add pairwise interaction (product) terms for the top-N most predictive features.

    Interactions are computed only for the highest-MI feature pairs to keep
    the feature space tractable. If ``target_col`` is not present, the first
    ``top_n`` columns are used.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_cols : list of str
        Candidate feature columns.
    top_n : int
        Number of top features to form interaction pairs from.
    target_col : str, optional
        Target column used to rank features by mutual information.

    Returns
    -------
    pd.DataFrame
        DataFrame extended with pairwise product interaction columns.
        New column names follow ``{col_a}__x__{col_b}``.
    """
    if target_col and target_col in df.columns:
        mi_scores = mutual_info_regression(
            df[feature_cols].fillna(0), df[target_col], random_state=42
        )
        top_features = [
            feature_cols[i]
            for i in np.argsort(mi_scores)[::-1][:top_n]
        ]
    else:
        top_features = feature_cols[:top_n]

    new_cols: dict = {}
    for i, col_a in enumerate(top_features):
        for col_b in top_features[i + 1 :]:
            key = f"{col_a}__x__{col_b}"
            new_cols[key] = (df[col_a] * df[col_b]).values

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Master feature engineering function
# ---------------------------------------------------------------------------


def engineer_features(
    df: pd.DataFrame,
    rolling_windows: List[int] = ROLLING_WINDOWS,
    neutralize: bool = True,
    add_interactions: bool = True,
    interaction_top_n: int = 10,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply the full feature engineering pipeline to a Numerai-style DataFrame.

    Pipeline steps:
        1. Identify numeric feature columns (excludes ``era`` and ``target``).
        2. Cross-sectional rank normalization within era.
        3. Rolling statistics (mean, std, skew) over multiple windows.
        4. Era-neutral projection.
        5. Pairwise interaction terms for top-MI features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame with an ``era`` column and numeric feature columns.
    rolling_windows : list of int
        Window sizes for rolling statistics.
    neutralize : bool
        Whether to apply era-neutral projection.
    add_interactions : bool
        Whether to add pairwise interaction terms.
    interaction_top_n : int
        Number of top features used to form interaction pairs.

    Returns
    -------
    df_out : pd.DataFrame
        Engineered DataFrame.
    feature_cols : list of str
        Names of all engineered feature columns (excludes ``era`` / ``target``).
    """
    meta_cols = {ERA_COL, TARGET_COL, "id"}
    base_feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Step 1: rank normalization
    df_out = rank_normalize_era(df.copy(), base_feature_cols)

    # Step 2: rolling features
    df_out = add_rolling_features(df_out, base_feature_cols, windows=rolling_windows)

    # Step 3: era neutralization (on base features only)
    if neutralize:
        df_out = era_neutralize(df_out, base_feature_cols)

    # Step 4: interaction terms
    if add_interactions:
        df_out = add_interaction_terms(
            df_out,
            base_feature_cols,
            top_n=interaction_top_n,
            target_col=TARGET_COL if TARGET_COL in df_out.columns else None,
        )

    feature_cols = [c for c in df_out.columns if c not in meta_cols]
    return df_out, feature_cols


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def select_features(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    n: int = 50,
    corr_threshold: float = 0.95,
) -> List[str]:
    """
    Select the top-N features using mutual information combined with a
    Pearson correlation filter to remove near-duplicate features.

    Algorithm:
        1. Compute mutual information between each feature and the target.
        2. Rank features by MI score descending.
        3. Greedily add features whose absolute Pearson correlation with
           all already-selected features is below ``corr_threshold``.
        4. Stop when N features are selected.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame containing the target column.
    target : str
        Name of the target column.
    n : int
        Maximum number of features to select.
    corr_threshold : float
        Features with |corr| >= this value to any already-selected feature
        are skipped.

    Returns
    -------
    list of str
        Selected feature column names, ordered by MI score.
    """
    meta_cols = {ERA_COL, TARGET_COL, "id"}
    candidate_cols = [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[candidate_cols].fillna(0.0)
    y = df[target].fillna(0.0)

    mi_scores = mutual_info_regression(X, y, random_state=42)
    ranked_cols = [candidate_cols[i] for i in np.argsort(mi_scores)[::-1]]

    selected: List[str] = []
    for col in ranked_cols:
        if len(selected) >= n:
            break
        if not selected:
            selected.append(col)
            continue
        corr_matrix = df[selected + [col]].corr()
        max_corr = corr_matrix[col].drop(col).abs().max()
        if max_corr < corr_threshold:
            selected.append(col)

    return selected
