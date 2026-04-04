"""
LightGBM Model for Numerai
--------------------------
Fastest and most commonly top-performing model in the Numerai tournament.
Implements multi-target training, era-sampling for training stability,
and optional Optuna hyperparameter search.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


# Best hyperparameters discovered through community research
# See: https://forum.numer.ai/t/hyperparameters-optimization-for-small-lgbm-models/6693
LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "num_leaves": 31,
    "colsample_bytree": 0.1,        # Low = more diverse trees = better generalization
    "subsample": 0.05,              # Era-level subsampling handled externally; row-level here
    "subsample_freq": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.01,
    "min_child_samples": 20,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
    "first_metric_only": True,
}

# High-capacity model for when you have time / RAM
LGBM_LARGE_PARAMS = {
    **LGBM_DEFAULT_PARAMS,
    "n_estimators": 6000,
    "learning_rate": 0.005,
    "max_depth": 6,
    "num_leaves": 63,
    "colsample_bytree": 0.08,
    "subsample": 0.05,
    "min_child_samples": 100,
}


class NumeraiLGBM:
    """
    LightGBM model wrapper for Numerai tournament.

    Supports:
    - Multi-target training (train on multiple targets, ensemble predictions)
    - Era-balanced sampling (treat eras as equal-weight data points)
    - Early stopping on validation set
    - Feature importance tracking
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        targets: Optional[List[str]] = None,
        era_col: str = "era",
        use_large: bool = False,
    ):
        self.params = params or (LGBM_LARGE_PARAMS if use_large else LGBM_DEFAULT_PARAMS)
        self.targets = targets or ["target"]
        self.era_col = era_col
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_cols: Optional[List[str]] = None

    def _era_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Per-row sample weights so each era contributes equally.
        Corrects for eras with different numbers of stocks.
        """
        era_sizes = df.groupby(self.era_col)[self.era_col].transform("count")
        return (1.0 / era_sizes).values.astype(np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 200,
    ) -> "NumeraiLGBM":
        """
        Train one model per target, with era-balanced sample weights.

        Args:
            train_df:              Training data with features, era, and target columns.
            feature_cols:          Feature column names.
            val_df:                Validation data for early stopping (optional).
            early_stopping_rounds: Stop if validation loss doesn't improve for N rounds.
            verbose_eval:          Print eval every N rounds.
        """
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols]
        w_train = self._era_sample_weights(train_df)

        for target in self.targets:
            if target not in train_df.columns:
                logger.warning(f"Target '{target}' not in training data, skipping.")
                continue

            y_train = train_df[target]

            logger.info(f"Training LightGBM on target: {target}")
            model = lgb.LGBMRegressor(**self.params)

            callbacks = [lgb.log_evaluation(period=verbose_eval)]

            if val_df is not None and target in val_df.columns:
                X_val = val_df[feature_cols]
                y_val = val_df[target]
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                )
            else:
                model.fit(X_train, y_train, sample_weight=w_train, callbacks=callbacks)

            self.models[target] = model
            logger.info(f"  Best iteration: {model.best_iteration_}, trees: {model.n_estimators_}")

        return self

    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate predictions.

        If multiple targets were trained, return the mean (target ensemble).
        """
        fc = feature_cols or self.feature_cols
        X = df[fc]

        preds_list = []
        for target, model in self.models.items():
            p = model.predict(X)
            # Gaussianize before averaging (keeps predictions comparable across targets)
            from scipy.stats import rankdata
            p_ranked = rankdata(p) / len(p)
            preds_list.append(p_ranked)

        return np.mean(preds_list, axis=0)

    def feature_importances(self, importance_type: str = "gain") -> pd.Series:
        """Return average feature importances across all target models."""
        if not self.models:
            raise RuntimeError("Model not trained yet.")
        fc = self.feature_cols
        importances = pd.DataFrame({
            target: model.feature_importances_
            for target, model in self.models.items()
        }, index=fc)
        return importances.mean(axis=1).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Pickle the model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        logger.info(f"Saved LightGBM model to {path}")

    @classmethod
    def load(cls, path: str) -> "NumeraiLGBM":
        """Load pickled model."""
        with open(path, "rb") as f:
            return pickle.load(f)


def tune_lgbm(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    era_col: str = "era",
    n_trials: int = 50,
    n_splits: int = 4,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """
    Bayesian hyperparameter optimization for LightGBM using Optuna.

    Args:
        n_trials:  Number of Optuna trials to run.
        timeout:   Max seconds for optimization.

    Returns:
        Best hyperparameters dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Run: pip install optuna")

    from ..utils.cross_validation import GroupedEraFold
    from ..utils.metrics import numerai_corr

    cv = GroupedEraFold(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 500, 8000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.3),
            "subsample": trial.suggest_float("subsample", 0.05, 0.3),
            "subsample_freq": 1,
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }

        scores = []
        for train_idx, val_idx in cv.split(train_df, era_col):
            tr = train_df.iloc[train_idx]
            va = train_df.iloc[val_idx]

            era_sizes = tr.groupby(era_col)[era_col].transform("count")
            weights = (1.0 / era_sizes).values

            model = lgb.LGBMRegressor(**params)
            model.fit(
                tr[feature_cols], tr[target_col],
                sample_weight=weights,
                callbacks=[lgb.log_evaluation(period=-1)],
            )
            preds = model.predict(va[feature_cols])
            preds_series = pd.Series(preds, index=va.index)
            score = va.groupby(era_col).apply(
                lambda g: numerai_corr(preds_series.loc[g.index], g[target_col])
            ).mean()
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info(f"Best CORR: {study.best_value:.5f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params
