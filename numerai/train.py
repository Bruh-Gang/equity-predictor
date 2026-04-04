"""
Numerai Training Pipeline using NumerBlox
-----------------------------------------
This pipeline implements an advanced scikit-learn structure via NumerBlox:
- Uses LGBMClassifier (extremely memory efficient on 50GB RAM).
- Incorporates CrossValEstimator with TimeSeriesSplit to fit multiple CV folds.
- Routes metadata (era, features) automatically.
- Integrates PredictionReducer, NumeraiEnsemble, and FeatureNeutralizer.
"""

import argparse
import logging
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from numerblox.meta import CrossValEstimator, make_meta_pipeline
from numerblox.ensemble import NumeraiEnsemble, PredictionReducer
from numerblox.neutralizers import FeatureNeutralizer

from configs.config import NumeraiConfig, DEFAULT_CONFIG
from utils.data_loader import download_data, load_numerframe_data
from utils.submission import format_predictions, save_predictions, upload_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/training.log", mode='w'),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Numerai Training Pipeline with NumerBlox")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--version", default="5.0", help="Dataset version")
    parser.add_argument("--submit", action="store_true", help="Upload predictions after training")
    parser.add_argument("--model-name", default=None, help="Numerai model name for submission")
    parser.add_argument("--neutralization", type=float, default=0.5,
                        help="Feature neutralization proportion (0.0-1.0, default: 0.5)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    return parser.parse_args()


def run_pipeline(cfg: NumeraiConfig, args) -> None:
    """Execute the full training pipeline using MetaPipeline."""
    start_time = time.time()
    Path("outputs").mkdir(exist_ok=True)

    # ===========================
    # 1. Download Data
    # ===========================
    logger.info("=" * 70)
    logger.info("STEP 1: Downloading & Loading Data")
    logger.info("=" * 70)

    # Use version from args or default to v5.0 (v5.0 supports int8 parquets by default with NumerBlox)
    data_path = download_data(cfg.data.data_dir, args.version)

    # ===========================
    # 2. Build MetaPipeline
    # ===========================
    logger.info("=" * 70)
    logger.info("STEP 2: Constructing MetaPipeline")
    logger.info("=" * 70)

    # We use LGBMClassifier for memory efficiency.
    # It natively handles int8 data from parquets beautifully.
    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=2**5-1,
        colsample_bytree=0.1,
        random_state=42,
        n_jobs=-1,
        device_type='cpu'
    )

    # CV Estimator ensures strict time-series splits
    crossval = CrossValEstimator(
        estimator=model, 
        cv=TimeSeriesSplit(n_splits=args.n_splits), 
        predict_func='predict_proba'
    )
    
    # Target conversion will result in 5 classes
    pred_rud = PredictionReducer(n_models=args.n_splits, n_classes=5)
    
    # Ensemble average
    ens = NumeraiEnsemble(donate_weighted=True)
    
    # Feature neutralization right inside the pipeline
    neut = FeatureNeutralizer(proportion=args.neutralization)
    
    # Combine into a single meta pipeline
    full_pipe = make_meta_pipeline(crossval, pred_rud, ens, neut)

    # ===========================
    # 3. Load Training Data & Fit
    # ===========================
    logger.info("=" * 70)
    logger.info("STEP 3: Fitting Pipeline on Training Data")
    logger.info("=" * 70)

    # Load only train logic, then release
    train_df = load_numerframe_data(data_path, "train.parquet")
    
    X, y = train_df.get_feature_target_pair(multi_target=False)
    # Convert targets (continuous 0.0-1.0) to 5 integer classes for classifier model
    y_int = (y * 4).astype(int)
    
    era_series = train_df.get_era_data
    features = train_df.get_feature_data

    logger.info(f"Shape of X: {X.shape}. Starting .fit() with {args.n_splits} folds...")
    
    full_pipe.fit(X, y_int, era_series=era_series)
    
    logger.info("Model fitting complete! Freeing Train Data from RAM...")
    
    # Explicit garbage collection to free memory before loading validation
    del X, y, y_int, era_series, features, train_df
    gc.collect()

    # ===========================
    # 4. Save the pipeline First!
    # ===========================
    logger.info("=" * 70)
    logger.info("STEP 4: Saving Completed Pipeline")
    logger.info("=" * 70)
    pipeline_path = "outputs/numerblox_meta_pipeline.pkl"
    joblib.dump(full_pipe, pipeline_path)
    logger.info(f"Saved full meta pipeline safely to: {pipeline_path}")

    # ===========================
    # 5. Evaluate on a Sub-Sample of Validation
    # ===========================
    logger.info("=" * 70)
    logger.info("STEP 5: Evaluating on Validation Slice")
    logger.info("=" * 70)

    val_df = load_numerframe_data(data_path, "validation.parquet")
    
    # Validation dataset is huge (4M rows). Running matrix inversion (Neutralization)
    # on all 4M rows takes astronomically long CPU time.
    # We will slice ONLY the last 10 eras for a quick metric check!
    unique_eras = val_df['era'].unique()
    safe_eras = unique_eras[-10:]
    val_df = val_df[val_df['era'].isin(safe_eras)].copy()
    logger.info(f"Sliced validation to {len(val_df)} rows to prevent memory crashes.")
    
    val_X, _ = val_df.get_feature_target_pair(multi_target=False)
    val_eras = val_df.get_era_data
    val_features = val_df.get_feature_data
    
    logger.info("Predicting safely on validation chunk...")
    val_preds = full_pipe.predict(val_X, era_series=val_eras, features=val_features)
    
    # Assign prediction back
    val_df['prediction'] = val_preds
    
    # Basic Evaluator hook
    try:
        from numerblox.evaluation import NumeraiClassicEvaluator
        from numerblox.prediction_loaders import ExamplePredictions
        
        example_preds_path = str(Path(args.data_dir) / f"{args.version}/validation_example_preds.parquet")
        if os.path.exists(example_preds_path):
            example_df = ExamplePredictions(example_preds_path).fit_transform(None)
            # Sync example predictions index with our sliced validation df
            val_df['example_preds'] = example_df.loc[val_df.index, 'prediction'].values
            
            evaluator = NumeraiClassicEvaluator()
            metrics = evaluator.full_evaluation(val_df, 
                                                example_col="example_preds", 
                                                pred_cols=["prediction"], 
                                                target_col="target")
            logger.info("\n=== VALIDATION METRICS ===")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.5f}")
    except Exception as e:
        logger.info(f"Metrics evaluation skipped/errored: {str(e)}")

    # Clear validation
    del val_df, val_X, val_eras, val_features
    gc.collect()

    # ===========================
    # Summary
    # ===========================
    elapsed = (time.time() - start_time) / 60
    logger.info("=" * 70)
    logger.info(f"TRAINING DONE! Total time: {elapsed:.1f} minutes")
    logger.info("=" * 70)

    return


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG

    # Apply overrides
    if args.model_name:
        cfg.submission.model_name = args.model_name

    run_pipeline(cfg, args)


if __name__ == "__main__":
    main()
