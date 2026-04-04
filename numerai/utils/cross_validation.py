"""
ERA-Aware Cross-Validation
--------------------------
Standard K-Fold is WRONG for Numerai due to:
1. Overlapping targets (20-day forward returns → eras within 4 weeks are correlated)
2. Temporal leakage (future eras must never appear in training)

This module implements proper era-based CV with embargo/purge periods.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, List, Optional


class EraFold:
    """
    Era-based time-series cross-validation with embargo.

    Each fold trains on past eras, tests on future eras.
    An embargo period skips eras immediately after training to prevent
    look-ahead bias from overlapping 20-day return targets.

    Args:
        n_splits:       Number of folds.
        embargo_size:   Number of eras to skip between train/test.
                        Use 5 for 20-day targets (5 weekly eras ≈ 1 month).
        gap:            Additional gap eras beyond embargo.
        max_train_size: Maximum number of train eras (None = use all history).
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_size: int = 5,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.embargo_size = embargo_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self,
        df: pd.DataFrame,
        era_col: str = "era",
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each fold.

        Uses walk-forward expanding window (more data over time).
        """
        eras = sorted(df[era_col].unique())
        n_eras = len(eras)
        era_to_idx = {era: i for i, era in enumerate(eras)}

        # Approximate fold size
        fold_size = n_eras // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError(f"Too few eras ({n_eras}) for {self.n_splits} splits.")

        for fold in range(self.n_splits):
            # Test fold occupies the next chunk of eras
            test_start_era_idx = (fold + 1) * fold_size
            test_end_era_idx = test_start_era_idx + fold_size

            # Embargo: skip eras just before test
            train_end_era_idx = test_start_era_idx - self.embargo_size - self.gap

            if self.max_train_size is not None:
                train_start_era_idx = max(0, train_end_era_idx - self.max_train_size)
            else:
                train_start_era_idx = 0

            if train_end_era_idx <= train_start_era_idx:
                continue  # Not enough history for this fold

            train_eras = set(eras[train_start_era_idx:train_end_era_idx])
            test_eras = set(eras[test_start_era_idx:test_end_era_idx])

            train_idx = df.index[df[era_col].isin(train_eras)].tolist()
            test_idx = df.index[df[era_col].isin(test_eras)].tolist()

            yield np.array(train_idx), np.array(test_idx)

    def get_splits_info(self, df: pd.DataFrame, era_col: str = "era") -> pd.DataFrame:
        """Return a summary DataFrame of fold sizes."""
        rows = []
        for i, (tr, te) in enumerate(self.split(df, era_col)):
            rows.append({
                "fold": i,
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_eras": df.loc[tr, era_col].nunique(),
                "test_eras": df.loc[te, era_col].nunique(),
            })
        return pd.DataFrame(rows)


class GroupedEraFold:
    """
    Group K-Fold variant: split eras into K equal groups.
    Each fold uses K-1 groups for training, 1 for testing.
    No temporal ordering guarantee — use for validation only, not time-series.
    Better for hyperparameter search when you have limited eras.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, df: pd.DataFrame, era_col: str = "era"):
        eras = np.array(sorted(df[era_col].unique()))
        era_groups = np.array_split(eras, self.n_splits)

        for i, test_era_group in enumerate(era_groups):
            train_eras = np.concatenate(
                [g for j, g in enumerate(era_groups) if j != i]
            )
            train_idx = df.index[df[era_col].isin(train_eras)].to_numpy()
            test_idx = df.index[df[era_col].isin(test_era_group)].to_numpy()
            yield train_idx, test_idx


def era_downsample(
    df: pd.DataFrame,
    era_col: str = "era",
    target_col: str = "target",
    n_eras: Optional[int] = None,
    n_rows_per_era: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Downsample training data to reduce memory and training time.

    Two modes:
    - n_eras: Randomly sample N eras from the dataset.
    - n_rows_per_era: Randomly sample N rows from each era.

    Tip: Numerai recommends treating eras as independent data points,
    so sampling eras is theoretically cleaner.
    """
    rng = np.random.default_rng(seed)
    eras = df[era_col].unique()

    if n_eras is not None:
        selected_eras = rng.choice(eras, size=min(n_eras, len(eras)), replace=False)
        return df[df[era_col].isin(selected_eras)].copy()

    if n_rows_per_era is not None:
        groups = []
        for era, group in df.groupby(era_col):
            if len(group) > n_rows_per_era:
                group = group.sample(n=n_rows_per_era, random_state=seed)
            groups.append(group)
        return pd.concat(groups)

    return df
