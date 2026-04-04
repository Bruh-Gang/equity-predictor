"""
Multi-Model Ensemble for Numerai
---------------------------------
Combines LightGBM, XGBoost, CatBoost, and Neural Network predictions.

Key principles:
1. Models must provide DIVERSE signals (different hyperparams, targets, architectures)
2. Average in RANK space (not raw prediction space) for robustness
3. Optionally learn optimal blend weights via correlation-based optimization
4. Apply feature neutralization post-ensemble for Sharpe improvement
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..utils.metrics import (
    neutralize_per_era,
    riskiest_features,
    validation_metrics,
    numerai_corr,
)

logger = logging.getLogger(__name__)


class NumeraiEnsemble:
    """
    Weighted ensemble of multiple Numerai models.

    Workflow:
    1. Train individual models
    2. Generate validation predictions from each model
    3. Optimize blend weights (maximize validation Sharpe)
    4. Apply feature neutralization on final ensemble predictions
    5. Generate live predictions for submission
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        neutralization_proportion: float = 0.5,
        n_riskiest_features: int = 50,
        era_col: str = "era",
    ):
        """
        Args:
            models:                      Dict of {name: model_object}.
            weights:                     Dict of {name: weight}. If None, equal weights.
            neutralization_proportion:   0.0 = none, 1.0 = full. 0.5 recommended.
            n_riskiest_features:         Number of riskiest features to neutralize against.
            era_col:                     Era column name.
        """
        self.models = models or {}
        self.weights = weights  # None = equal weights
        self.neutralization_proportion = neutralization_proportion
        self.n_riskiest_features = n_riskiest_features
        self.era_col = era_col
        self._riskiest_feats: Optional[List[str]] = None

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        if self.weights is None:
            self.weights = {}
        self.weights[name] = weight

    def _get_weights(self) -> Dict[str, float]:
        """Return normalized weights."""
        names = list(self.models.keys())
        if self.weights is None:
            w = {n: 1.0 / len(names) for n in names}
        else:
            total = sum(self.weights.get(n, 1.0) for n in names)
            w = {n: self.weights.get(n, 1.0) / total for n in names}
        return w

    def predict_all(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Get predictions from all models.

        Returns DataFrame with one column per model, values in [0,1] rank space.
        """
        preds = {}
        for name, model in self.models.items():
            logger.info(f"Generating predictions from: {name}")
            p = model.predict(df, feature_cols)
            # Ensure rank-normalized [0,1]
            preds[name] = rankdata(p) / len(p)
        return pd.DataFrame(preds, index=df.index)

    def blend(
        self,
        pred_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Weighted average of model predictions in rank space.

        After averaging, re-rank to maintain uniform distribution.
        """
        w = weights or self._get_weights()
        blended = sum(pred_df[name] * w.get(name, 0.0) for name in pred_df.columns)
        # Re-rank to [0,1]
        blended = pd.Series(rankdata(blended) / len(blended), index=pred_df.index)
        return blended

    def optimize_weights(
        self,
        val_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        target_col: str = "target",
        n_trials: int = 500,
    ) -> Dict[str, float]:
        """
        Optimize blend weights to maximize validation Sharpe using random search.

        This is a simple but effective approach — Numerai community research shows
        that elaborately tuned weights rarely outperform equal weighting out-of-sample.
        Random search over Dirichlet-sampled weights is a solid middle ground.
        """
        model_names = list(pred_df.columns)
        n_models = len(model_names)
        best_sharpe = -np.inf
        best_weights = {n: 1.0 / n_models for n in model_names}

        rng = np.random.default_rng(42)
        for _ in range(n_trials):
            # Dirichlet sample for valid weight simplex
            alpha = rng.dirichlet(np.ones(n_models))
            w = dict(zip(model_names, alpha))
            blended = self.blend(pred_df, w)
            val_df = val_df.copy()
            val_df["_ensemble_pred"] = blended.values
            metrics = validation_metrics(val_df, "_ensemble_pred", target_col, self.era_col)
            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_weights = w

        logger.info(f"Best validation Sharpe: {best_sharpe:.4f}")
        logger.info(f"Best weights: { {k: round(v,3) for k,v in best_weights.items()} }")
        self.weights = best_weights
        return best_weights

    def find_riskiest_features(
        self,
        train_df: pd.DataFrame,
        pred_col: str,
        feature_cols: List[str],
    ) -> List[str]:
        """Find and cache the riskiest features for neutralization."""
        self._riskiest_feats = riskiest_features(
            train_df, pred_col, feature_cols, self.era_col, n=self.n_riskiest_features
        )
        logger.info(f"Found {len(self._riskiest_feats)} riskiest features for neutralization.")
        return self._riskiest_feats

    def neutralize(
        self,
        df: pd.DataFrame,
        pred_col: str,
        feature_cols: List[str],
        risky_features: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Apply per-era feature neutralization.

        Args:
            risky_features: If None, uses all feature_cols (full neutralization).
                            Pass self._riskiest_feats for partial neutralization.
        """
        neut_cols = risky_features or self._riskiest_feats or feature_cols
        logger.info(f"Neutralizing against {len(neut_cols)} features "
                    f"(proportion={self.neutralization_proportion})")
        return neutralize_per_era(
            df, pred_col, neut_cols, self.neutralization_proportion, self.era_col
        )

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        apply_neutralization: bool = True,
        risky_features: Optional[List[str]] = None,
        pred_col_name: str = "prediction",
    ) -> pd.DataFrame:
        """
        Full prediction pipeline: predict → blend → neutralize.

        Returns DataFrame with 'id' and 'prediction' columns ready for submission.
        """
        # Step 1: Get all model predictions
        pred_df = self.predict_all(df, feature_cols)

        # Step 2: Blend
        blended = self.blend(pred_df)
        result_df = df[[self.era_col]].copy()
        result_df[pred_col_name] = blended.values

        # Step 3: Feature neutralization (if validation/train — needs era column)
        if apply_neutralization and self.neutralization_proportion > 0:
            if self.era_col in df.columns:
                combined = df[feature_cols + [self.era_col]].copy()
                combined[pred_col_name] = result_df[pred_col_name].values
                neutralized = self.neutralize(
                    combined, pred_col_name, feature_cols, risky_features
                )
                result_df[pred_col_name] = neutralized.values
            else:
                logger.warning("No era column in live data — skipping per-era neutralization.")

        return result_df

    def evaluate(
        self,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        apply_neutralization: bool = True,
    ) -> Dict[str, float]:
        """Evaluate ensemble on validation set. Returns metrics dict."""
        pred_df_all = self.predict_all(val_df, feature_cols)
        blended = self.blend(pred_df_all)

        val_df = val_df.copy()
        val_df["_raw"] = blended.values

        if apply_neutralization and self.neutralization_proportion > 0 and self._riskiest_feats:
            combined = val_df[feature_cols + [self.era_col]].copy()
            combined["_raw"] = blended.values
            val_df["_neutralized"] = self.neutralize(
                combined, "_raw", feature_cols, self._riskiest_feats
            ).values

        raw_metrics = validation_metrics(val_df, "_raw", target_col, self.era_col)
        metrics = {"raw_" + k: v for k, v in raw_metrics.items()}

        if "_neutralized" in val_df.columns:
            neut_metrics = validation_metrics(val_df, "_neutralized", target_col, self.era_col)
            metrics.update({"neut_" + k: v for k, v in neut_metrics.items()})

        return metrics
