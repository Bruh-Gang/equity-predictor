"""
Numerai Scoring Metrics
-----------------------
Implements CORR (Numerai Correlation), MMC-like diagnostics,
Sharpe, Max Drawdown, and Feature Neutralization utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Core Numerai Correlation (official scoring metric)
# ---------------------------------------------------------------------------

def gaussianize(predictions: pd.Series) -> pd.Series:
    """Rank-based gaussianization used in official Numerai CORR."""
    ranked = predictions.rank(pct=True, method="average")
    # Clip to avoid infinite values from inverse CDF
    ranked = ranked.clip(1e-4, 1 - 1e-4)
    return pd.Series(stats.norm.ppf(ranked), index=predictions.index)


def tie_broken_rank(predictions: pd.Series) -> pd.Series:
    """Rank with random tie-breaking (Numerai standard)."""
    return predictions.rank(pct=True, method="first")


def numerai_corr(predictions: pd.Series, targets: pd.Series) -> float:
    """
    Official Numerai correlation metric.

    Steps:
    1. Rank predictions with tie-breaking
    2. Gaussianize ranked predictions
    3. Raise to power 1.5 (preserving sign)
    4. Raise targets to power 1.5 (preserving sign)
    5. Pearson correlation
    """
    ranked_preds = tie_broken_rank(predictions)
    gaussianized = gaussianize(ranked_preds)
    powered_preds = np.sign(gaussianized) * np.abs(gaussianized) ** 1.5

    centered_targets = targets - targets.mean()
    powered_targets = np.sign(centered_targets) * np.abs(centered_targets) ** 1.5

    return np.corrcoef(powered_preds, powered_targets)[0, 1]


def spearman_corr(predictions: pd.Series, targets: pd.Series) -> float:
    """Spearman rank correlation (fast approximation)."""
    return predictions.corr(targets, method="spearman")


# ---------------------------------------------------------------------------
# Era-level metrics
# ---------------------------------------------------------------------------

def per_era_corr(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    era_col: str = "era",
    metric: str = "numerai",
) -> pd.Series:
    """
    Compute per-era correlation scores.

    Args:
        metric: 'numerai' (official), 'spearman', or 'pearson'
    """
    corr_fn = {
        "numerai": numerai_corr,
        "spearman": spearman_corr,
        "pearson": lambda p, t: p.corr(t, method="pearson"),
    }[metric]

    return df.groupby(era_col).apply(
        lambda g: corr_fn(g[pred_col], g[target_col])
    )


def mean_std_sharpe(era_scores: pd.Series, annualize: bool = True) -> Dict[str, float]:
    """
    Compute mean, std, and Sharpe of era scores.
    Annualize assumes ~50 eras/year (weekly eras).
    """
    mean = era_scores.mean()
    std = era_scores.std(ddof=1)
    sharpe = mean / std if std > 0 else 0.0
    if annualize:
        sharpe *= np.sqrt(52)
    return {"mean": mean, "std": std, "sharpe": sharpe}


def max_drawdown(era_scores: pd.Series) -> float:
    """Maximum drawdown of cumulative era scores."""
    cumulative = era_scores.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    return drawdown.min()


def smart_sharpe(era_scores: pd.Series) -> float:
    """
    Smart Sharpe: penalizes non-Gaussian distributions.
    A favorite metric in the Numerai community.
    """
    mean = era_scores.mean()
    std = era_scores.std(ddof=1)
    skew = era_scores.skew()
    kurt = era_scores.kurtosis()
    smart = mean / std * (1 - 0.5 * skew * (mean / std) - (kurt / 4) * (mean / std) ** 2)
    return smart * np.sqrt(52)


def validation_metrics(
    val_df: pd.DataFrame,
    pred_col: str,
    target_col: str = "target",
    era_col: str = "era",
) -> Dict[str, float]:
    """
    Comprehensive validation metrics.

    Returns dict with: mean_corr, std_corr, sharpe, smart_sharpe, max_drawdown,
    pct_positive, feature_neutral_corr (if features available).
    """
    era_scores = per_era_corr(val_df, pred_col, target_col, era_col)
    ss = mean_std_sharpe(era_scores)

    return {
        "mean_corr": ss["mean"],
        "std_corr": ss["std"],
        "sharpe": ss["sharpe"],
        "smart_sharpe": smart_sharpe(era_scores),
        "max_drawdown": max_drawdown(era_scores),
        "pct_positive_eras": (era_scores > 0).mean(),
        "num_eras": len(era_scores),
    }


# ---------------------------------------------------------------------------
# Feature Neutralization
# ---------------------------------------------------------------------------

def neutralize(
    predictions: pd.Series,
    neutralizers: pd.DataFrame,
    proportion: float = 1.0,
) -> pd.Series:
    """
    Neutralize predictions against a set of features (OLS projection).

    Removes linear dependency of predictions on neutralizer features.
    Standard technique to improve Numerai Sharpe and reduce feature exposure.

    Args:
        predictions:  Model predictions (already era-specific).
        neutralizers: Feature matrix to neutralize against (same index).
        proportion:   0.0 = no neutralization, 1.0 = full neutralization.
                      0.5 is a common sweet spot (official example uses 0.5).

    Returns:
        Neutralized predictions, re-ranked 0→1.
    """
    # Standardize neutralizers
    neut = neutralizers.values.astype(np.float64)
    preds = predictions.values.astype(np.float64).reshape(-1, 1)

    # OLS: project predictions onto feature space
    # proj = X @ (X'X)^{-1} @ X' @ y
    try:
        scores = neut - neut.mean(axis=0)
        exposures = scores.T @ preds
        # Solve least squares
        beta = np.linalg.lstsq(scores.T @ scores, exposures, rcond=None)[0]
        neutralized = preds - proportion * (scores @ beta)
    except np.linalg.LinAlgError:
        return predictions  # fallback: no neutralization

    result = neutralized.ravel()
    # Rank-normalize back to [0, 1]
    result = pd.Series(result, index=predictions.index)
    result = result.rank(pct=True, method="average")
    return result


def neutralize_per_era(
    df: pd.DataFrame,
    pred_col: str,
    neutralizer_cols: list,
    proportion: float = 0.5,
    era_col: str = "era",
) -> pd.Series:
    """
    Apply feature neutralization independently within each era.
    This is the correct way — neutralization must be era-local.
    """
    neutralized = pd.Series(index=df.index, dtype=np.float64)
    for era, group in df.groupby(era_col):
        preds = group[pred_col]
        neut_features = group[neutralizer_cols]
        neutralized.loc[group.index] = neutralize(preds, neut_features, proportion)
    return neutralized


# ---------------------------------------------------------------------------
# Feature Exposure
# ---------------------------------------------------------------------------

def feature_exposure(
    df: pd.DataFrame,
    pred_col: str,
    feature_cols: list,
    era_col: str = "era",
) -> pd.DataFrame:
    """
    Compute per-era correlation between predictions and each feature.
    High exposure = model relies heavily on that feature (risky).
    """
    exposures = df.groupby(era_col).apply(
        lambda g: g[feature_cols].corrwith(g[pred_col])
    )
    return exposures


def max_feature_exposure(
    df: pd.DataFrame,
    pred_col: str,
    feature_cols: list,
    era_col: str = "era",
) -> pd.Series:
    """Max absolute per-era feature exposure per feature."""
    exp = feature_exposure(df, pred_col, feature_cols, era_col)
    return exp.abs().max()


def riskiest_features(
    train_df: pd.DataFrame,
    pred_col: str,
    feature_cols: list,
    era_col: str = "era",
    n: int = 50,
) -> list:
    """
    Find the N features whose correlation with predictions changes most across eras.
    These are the 'risky' features to neutralize against.
    """
    era_feature_corrs = train_df.groupby(era_col).apply(
        lambda g: g[feature_cols].corrwith(g[pred_col])
    )
    # Largest variation = riskiest
    return era_feature_corrs.std().nlargest(n).index.tolist()
