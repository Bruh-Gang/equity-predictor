"""
models.py
---------
Model classes for equity return prediction.

Provides XGBoostModel and LightGBMModel wrappers with a unified
fit / predict / cross_validate interface, and a StackedEnsemble that
trains a Ridge meta-learner on out-of-fold (OOF) predictions.

Evaluation via evaluate_model() returns Numerai-relevant metrics:
Pearson/Spearman correlation, era-wise Sharpe ratio, and max drawdown.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

XGBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [200, 500, 800],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.005, 0.01, 0.05],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

LIGHTGBM_DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "num_leaves": 31,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "objective": "regression",
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

LIGHTGBM_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [200, 500, 800],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.005, 0.01, 0.05],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spearman correlation between two arrays."""
    return float(spearmanr(y_true, y_pred)[0])


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Pearson correlation between two arrays."""
    return float(pearsonr(y_true, y_pred)[0])


# ---------------------------------------------------------------------------
# XGBoostModel
# ---------------------------------------------------------------------------


class XGBoostModel:
    """
    XGBoost wrapper with fit, predict, and era-aware cross-validation.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters. Any key from XGBOOST_DEFAULT_PARAMS
        is accepted. Missing keys fall back to defaults.

    Attributes
    ----------
    model : xgb.XGBRegressor
        Underlying XGBoost estimator.
    feature_importances_ : np.ndarray or None
        Feature importances (gain) after fitting.
    """

    param_grid = XGBOOST_PARAM_GRID

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = {**XGBOOST_DEFAULT_PARAMS, **(params or {})}
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = False,
    ) -> "XGBoostModel":
        """
        Train the XGBoost model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        eval_set : list of (X_val, y_val) tuples, optional
        verbose : bool
            Whether to print training progress.

        Returns
        -------
        self
        """
        fit_kwargs: Dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = verbose
        self.model.fit(X, y, **fit_kwargs)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        return self.model.predict(X)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eras: Optional[np.ndarray] = None,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run K-fold cross-validation, respecting era boundaries if provided.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        eras : array-like of shape (n_samples,), optional
            Era labels. When provided, folds are constructed so that
            no era spans two folds (era-grouped CV).
        n_folds : int
            Number of folds.

        Returns
        -------
        dict with keys:
            - ``oof_predictions``: np.ndarray of OOF predictions
            - ``val_spearman``: list of per-fold Spearman correlations
            - ``val_pearson``: list of per-fold Pearson correlations
            - ``mean_spearman``: float
            - ``mean_pearson``: float
            - ``feature_importances``: np.ndarray (mean across folds)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        oof_preds = np.zeros(len(y))
        val_spearman: List[float] = []
        val_pearson: List[float] = []
        importances: List[np.ndarray] = []

        if eras is not None:
            fold_indices = _era_kfold_indices(eras, n_folds)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(X))

        for train_idx, val_idx in fold_indices:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = XGBoostModel(self.params)
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)
            oof_preds[val_idx] = preds

            val_spearman.append(_spearman(y_val, preds))
            val_pearson.append(_pearson(y_val, preds))
            if fold_model.feature_importances_ is not None:
                importances.append(fold_model.feature_importances_)

        self.feature_importances_ = np.mean(importances, axis=0) if importances else None

        return {
            "oof_predictions": oof_preds,
            "val_spearman": val_spearman,
            "val_pearson": val_pearson,
            "mean_spearman": float(np.mean(val_spearman)),
            "mean_pearson": float(np.mean(val_pearson)),
            "feature_importances": self.feature_importances_,
        }

    def save(self, path: str) -> None:
        """Serialize model to disk using joblib."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "XGBoostModel":
        """Load a previously saved XGBoostModel."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# LightGBMModel
# ---------------------------------------------------------------------------


class LightGBMModel:
    """
    LightGBM wrapper with fit, predict, and era-aware cross-validation.

    Parameters
    ----------
    params : dict, optional
        Override default hyperparameters.

    Attributes
    ----------
    model : lgb.LGBMRegressor
    feature_importances_ : np.ndarray or None
    """

    param_grid = LIGHTGBM_PARAM_GRID

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = {**LIGHTGBM_DEFAULT_PARAMS, **(params or {})}
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = False,
    ) -> "LightGBMModel":
        """
        Train the LightGBM model.

        Parameters
        ----------
        X : array-like
        y : array-like
        eval_set : list of (X_val, y_val), optional
        verbose : bool

        Returns
        -------
        self
        """
        fit_kwargs: Dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        self.model.fit(X, y, **fit_kwargs)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray
        """
        return self.model.predict(X)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eras: Optional[np.ndarray] = None,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run K-fold cross-validation with optional era grouping.

        Parameters mirror XGBoostModel.cross_validate.

        Returns
        -------
        dict with keys identical to XGBoostModel.cross_validate.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        oof_preds = np.zeros(len(y))
        val_spearman: List[float] = []
        val_pearson: List[float] = []
        importances: List[np.ndarray] = []

        if eras is not None:
            fold_indices = _era_kfold_indices(eras, n_folds)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(X))

        for train_idx, val_idx in fold_indices:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = LightGBMModel(self.params)
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_val)
            oof_preds[val_idx] = preds

            val_spearman.append(_spearman(y_val, preds))
            val_pearson.append(_pearson(y_val, preds))
            if fold_model.feature_importances_ is not None:
                importances.append(fold_model.feature_importances_)

        self.feature_importances_ = np.mean(importances, axis=0) if importances else None

        return {
            "oof_predictions": oof_preds,
            "val_spearman": val_spearman,
            "val_pearson": val_pearson,
            "mean_spearman": float(np.mean(val_spearman)),
            "mean_pearson": float(np.mean(val_pearson)),
            "feature_importances": self.feature_importances_,
        }

    def save(self, path: str) -> None:
        """Serialize model to disk using joblib."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "LightGBMModel":
        """Load a previously saved LightGBMModel."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Stacked Ensemble
# ---------------------------------------------------------------------------


class StackedEnsemble:
    """
    Two-layer stacked ensemble with a Ridge regression meta-learner.

    Base models are trained with cross-validation; their out-of-fold (OOF)
    predictions are used as features to train the meta-learner. At inference
    time, all base models predict on new data, and the meta-learner blends
    those predictions.

    Parameters
    ----------
    base_models : list
        List of instantiated model objects (XGBoostModel / LightGBMModel).
    meta_alpha : float
        Ridge regularization strength for the meta-learner.
    n_folds : int
        Number of CV folds used for OOF generation.

    Attributes
    ----------
    meta_learner : Ridge
    oof_matrix_ : np.ndarray, shape (n_train, n_base_models)
    cv_results_ : list of dict
        Per-base-model cross-validation output dicts.
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_alpha: float = 1.0,
        n_folds: int = 5,
    ) -> None:
        self.base_models = base_models
        self.meta_alpha = meta_alpha
        self.n_folds = n_folds
        self.meta_learner = Ridge(alpha=meta_alpha, fit_intercept=True)
        self.oof_matrix_: Optional[np.ndarray] = None
        self.cv_results_: List[Dict[str, Any]] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eras: Optional[np.ndarray] = None,
    ) -> "StackedEnsemble":
        """
        Train base models via CV to produce OOF predictions, then fit
        the Ridge meta-learner on those OOF predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        eras : array-like, optional
            Era labels for era-grouped cross-validation.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        oof_matrix = np.zeros((n, len(self.base_models)))

        for i, model in enumerate(self.base_models):
            print(f"  Training base model {i + 1}/{len(self.base_models)}: {type(model).__name__}")
            cv_result = model.cross_validate(X, y, eras=eras, n_folds=self.n_folds)
            oof_matrix[:, i] = cv_result["oof_predictions"]
            self.cv_results_.append(cv_result)
            print(
                f"    → Mean Spearman: {cv_result['mean_spearman']:.4f} | "
                f"Mean Pearson: {cv_result['mean_pearson']:.4f}"
            )
            # Refit on full training data
            model.fit(X, y)

        self.oof_matrix_ = oof_matrix
        self.meta_learner.fit(oof_matrix, y)
        print("  Meta-learner fitted on OOF predictions.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions on new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Weighted blend of base model predictions via the meta-learner.
        """
        X = np.asarray(X)
        base_preds = np.column_stack([m.predict(X) for m in self.base_models])
        return self.meta_learner.predict(base_preds)

    def save(self, path: str) -> None:
        """Serialize the full ensemble (base models + meta-learner) to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "StackedEnsemble":
        """Load a previously saved StackedEnsemble."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    predictions: np.ndarray,
    targets: np.ndarray,
    eras: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a suite of Numerai-relevant evaluation metrics.

    Parameters
    ----------
    predictions : np.ndarray, shape (n_samples,)
        Model output scores.
    targets : np.ndarray, shape (n_samples,)
        Ground-truth return labels.
    eras : np.ndarray, optional
        Era labels. Required for Sharpe and max drawdown computation.

    Returns
    -------
    dict with keys:
        - ``pearson_corr``: float — global Pearson r
        - ``spearman_corr``: float — global Spearman r
        - ``sharpe_ratio``: float — mean / std of era-wise Spearman (or NaN)
        - ``max_drawdown``: float — worst cumulative era-corr drawdown (or NaN)
        - ``era_correlations``: list of float — per-era Spearman values
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    pearson = float(pearsonr(predictions, targets)[0])
    spearman = float(spearmanr(predictions, targets)[0])

    era_correlations: List[float] = []
    sharpe = float("nan")
    max_dd = float("nan")

    if eras is not None:
        unique_eras = pd.Series(eras).unique()
        for era in unique_eras:
            mask = eras == era
            if mask.sum() < 2:
                continue
            era_corr = float(spearmanr(predictions[mask], targets[mask])[0])
            era_correlations.append(era_corr)

        if len(era_correlations) > 1:
            corr_arr = np.array(era_correlations)
            sharpe = float(np.mean(corr_arr) / (np.std(corr_arr) + 1e-8))
            # Max drawdown on cumulative correlations
            cumulative = np.cumsum(corr_arr)
            peak = np.maximum.accumulate(cumulative)
            drawdown = cumulative - peak
            max_dd = float(drawdown.min())

    return {
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "era_correlations": era_correlations,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _era_kfold_indices(
    eras: np.ndarray, n_folds: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build train/val index pairs where each era is entirely in one fold.

    Eras are assigned to folds in round-robin order to produce balanced folds.

    Parameters
    ----------
    eras : np.ndarray
        Era label per sample.
    n_folds : int

    Returns
    -------
    list of (train_indices, val_indices) tuples
    """
    eras = np.asarray(eras)
    unique_eras = np.unique(eras)
    era_to_fold = {era: i % n_folds for i, era in enumerate(unique_eras)}
    sample_folds = np.array([era_to_fold[e] for e in eras])

    all_indices = np.arange(len(eras))
    fold_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_folds):
        val_mask = sample_folds == fold_id
        train_idx = all_indices[~val_mask]
        val_idx = all_indices[val_mask]
        fold_pairs.append((train_idx, val_idx))

    return fold_pairs
