"""
XGBoost Model for Numerai
-------------------------
Strong baseline with excellent regularization.
Diversity from LightGBM makes it valuable in the ensemble.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


XGB_DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",         # GPU: set to "gpu_hist"
    "device": "cpu",               # Change to "cuda" for GPU
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 5,
    "colsample_bytree": 0.10,
    "subsample": 0.1,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

XGB_LARGE_PARAMS = {
    **XGB_DEFAULT_PARAMS,
    "n_estimators": 5000,
    "learning_rate": 0.005,
    "max_depth": 6,
    "colsample_bytree": 0.08,
    "subsample": 0.07,
    "min_child_weight": 50,
}


class NumeraiXGB:
    """XGBoost model for Numerai with multi-target support."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        targets: Optional[List[str]] = None,
        era_col: str = "era",
        use_large: bool = False,
    ):
        self.params = params or (XGB_LARGE_PARAMS if use_large else XGB_DEFAULT_PARAMS)
        self.targets = targets or ["target"]
        self.era_col = era_col
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_cols: Optional[List[str]] = None

    def _era_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        era_sizes = df.groupby(self.era_col)[self.era_col].transform("count")
        return (1.0 / era_sizes).values.astype(np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 50,
        verbose: int = 200,
    ) -> "NumeraiXGB":
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols]
        w_train = self._era_sample_weights(train_df)

        for target in self.targets:
            if target not in train_df.columns:
                logger.warning(f"Target '{target}' not found, skipping.")
                continue

            y_train = train_df[target]
            logger.info(f"Training XGBoost on target: {target}")

            model = xgb.XGBRegressor(**self.params)

            if val_df is not None and target in val_df.columns:
                X_val = val_df[feature_cols]
                y_val = val_df[target]
                model.set_params(early_stopping_rounds=early_stopping_rounds)
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=verbose,
                )
            else:
                model.fit(X_train, y_train, sample_weight=w_train, verbose=verbose)

            self.models[target] = model
            logger.info(f"  Done. Best iteration: {model.best_iteration}")

        return self

    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        fc = feature_cols or self.feature_cols
        X = df[fc]
        preds_list = []
        for target, model in self.models.items():
            p = model.predict(X)
            p_ranked = rankdata(p) / len(p)
            preds_list.append(p_ranked)
        return np.mean(preds_list, axis=0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> "NumeraiXGB":
        with open(path, "rb") as f:
            return pickle.load(f)
