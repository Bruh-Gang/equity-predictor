"""
CatBoost Model for Numerai
--------------------------
Excellent regularization and handles overfitting well.
Symmetric trees → fast inference, good diversity from LightGBM/XGBoost.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


CATBOOST_DEFAULT_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.01,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "colsample_bylevel": 0.1,
    "min_data_in_leaf": 20,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 200,
    "task_type": "CPU",           # Set to "GPU" if available
    "thread_count": -1,
}

CATBOOST_LARGE_PARAMS = {
    **CATBOOST_DEFAULT_PARAMS,
    "iterations": 5000,
    "learning_rate": 0.005,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "colsample_bylevel": 0.08,
    "min_data_in_leaf": 100,
}


class NumeraiCatBoost:
    """CatBoost model for Numerai."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        targets: Optional[List[str]] = None,
        era_col: str = "era",
        use_large: bool = False,
    ):
        try:
            from catboost import CatBoostRegressor
            self._cb_cls = CatBoostRegressor
        except ImportError:
            raise ImportError("Run: pip install catboost")

        self.params = params or (CATBOOST_LARGE_PARAMS if use_large else CATBOOST_DEFAULT_PARAMS)
        self.targets = targets or ["target"]
        self.era_col = era_col
        self.models: Dict[str, Any] = {}
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
    ) -> "NumeraiCatBoost":
        from catboost import Pool

        self.feature_cols = feature_cols
        w_train = self._era_sample_weights(train_df)

        for target in self.targets:
            if target not in train_df.columns:
                logger.warning(f"Target '{target}' not found, skipping.")
                continue

            logger.info(f"Training CatBoost on target: {target}")
            train_pool = Pool(train_df[feature_cols], train_df[target], weight=w_train)

            params = dict(self.params)
            model = self._cb_cls(**params)

            if val_df is not None and target in val_df.columns:
                val_pool = Pool(val_df[feature_cols], val_df[target])
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:
                model.fit(train_pool)

            self.models[target] = model
            logger.info(f"  Done. Best iteration: {model.best_iteration_}")

        return self

    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        fc = feature_cols or self.feature_cols
        preds_list = []
        for target, model in self.models.items():
            p = model.predict(df[fc])
            p_ranked = rankdata(p) / len(p)
            preds_list.append(p_ranked)
        return np.mean(preds_list, axis=0)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> "NumeraiCatBoost":
        with open(path, "rb") as f:
            return pickle.load(f)
