"""
Numerai Pipeline Configuration
-------------------------------
Single place to configure everything. Edit this file to adjust your strategy.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    data_dir: str = "data"
    version: str = "v4.2"           # Latest dataset version
    feature_set: str = "medium"     # 'small', 'medium', 'all'
    # Main target — engineered by Numerai to match their hedge fund strategy
    target: str = "target_cyrus_v4_20"   # Current recommended target
    # Auxiliary targets for ensemble (train a model on each, then average)
    aux_targets: List[str] = field(default_factory=lambda: [
        "target_victor_v4_20",
        "target_ralph_v4_20",
        "target_waldo_v4_20",
        "target_jerome_v4_20",
        "target_evelyn_v4_20",
    ])
    int8: bool = True               # Load int8 parquet (faster, less RAM)


@dataclass
class ModelConfig:
    # Which models to include in the ensemble
    use_lgbm: bool = True
    use_xgb: bool = True
    use_catboost: bool = True
    use_nn: bool = True

    # Use large (slower but more powerful) model configs
    use_large_models: bool = True   # Recommended with 50GB RAM

    # LightGBM
    lgbm_params: Optional[Dict[str, Any]] = None  # None = use defaults

    # XGBoost
    xgb_params: Optional[Dict[str, Any]] = None

    # CatBoost
    catboost_params: Optional[Dict[str, Any]] = None

    # Neural Network
    nn_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    nn_dropout: float = 0.1
    nn_epochs: int = 100
    nn_batch_size: int = 8192
    nn_learning_rate: float = 1e-3
    nn_swa_start: int = 50

    # Ensemble weights (None = optimize on validation)
    ensemble_weights: Optional[Dict[str, float]] = None


@dataclass
class EnsembleConfig:
    # Feature neutralization (key to Sharpe improvement)
    neutralization_proportion: float = 0.5   # Community-validated sweet spot
    n_riskiest_features: int = 50            # Features to neutralize against

    # Blend weight optimization
    optimize_weights: bool = True
    weight_optimization_trials: int = 500


@dataclass
class TrainingConfig:
    # Cross-validation
    n_cv_splits: int = 5
    embargo_eras: int = 5      # Skip 5 eras after train (avoid 20-day target overlap)

    # Early stopping
    early_stopping_rounds: int = 100

    # Hyperparameter tuning
    run_hparam_tuning: bool = False    # Set True for full HPO (very slow)
    n_optuna_trials: int = 50
    hparam_timeout: int = 3600         # 1 hour per model

    # Reproducibility
    seed: int = 42


@dataclass
class SubmissionConfig:
    model_name: str = "YOUR_MODEL_NAME"   # <-- Set this to your Numerai model name
    output_dir: str = "outputs"
    auto_upload: bool = False              # Set True to auto-submit after training


@dataclass
class NumeraiConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)


# ============================
# DEFAULT CONFIGURATION
# ============================
# This is what you'll typically use. Edit to taste.
DEFAULT_CONFIG = NumeraiConfig()
