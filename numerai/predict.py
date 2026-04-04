"""
Numerai Live Prediction (Weekly Submission) with NumerBlox
----------------------------------------------------------
Loads the trained MetaPipeline and generates predictions for the live round.
Outputs are neutrally routed directly in the pipeline, simplifying inference.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from numerblox.download import NumeraiClassicDownloader
from numerblox.numerframe import create_numerframe
from numerblox.submission import NumeraiClassicSubmitter
from numerblox.misc import Key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_live(
    model_name: str,
    data_dir: str = "data",
    version: str = "5.0",
    auto_submit: bool = False,
    outputs_dir: str = "outputs",
) -> pd.DataFrame:
    """Load models, generate live predictions using the pipeline, optionally submit."""

    pipeline_path = Path(outputs_dir) / "numerblox_meta_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline not found at {pipeline_path}. Run train.py first.")

    # 1. Download live data
    logger.info("Downloading live data...")
    downloader = NumeraiClassicDownloader(data_dir)
    downloader.download_live_data("current_round", version=version)
    
    live_path = Path(data_dir) / "current_round" / "live.parquet"

    # 2. Load live data as memory efficient NumerFrame
    logger.info("Loading live data...")
    live_df = create_numerframe(file_path=str(live_path))
    logger.info(f"Live data: {len(live_df)} rows")

    # 3. Predict directly from the meta-pipeline
    logger.info("Loading pipeline and generating predictions...")
    full_pipe = joblib.load(pipeline_path)
    
    live_X = live_df.get_feature_data
    live_eras = live_df.get_era_data
    live_features = live_df.get_feature_data
    
    logger.info("Unpacking MetaPipeline components to execute securely...")
    cve, pr, ens, fn = [s[1] for s in full_pipe.steps]
    
    logger.info("1. Executing CrossValEstimator models...")
    p1 = cve.transform(live_X)
    
    logger.info("2. Executing PredictionReducer...")
    p2 = pr.transform(p1)
    
    logger.info("3. Executing NumeraiEnsemble...")
    p3 = ens.transform(p2, era_series=live_eras)
    
    logger.info("4. Executing FeatureNeutralizer...")
    preds = fn.predict(p3, features=live_features, era_series=live_eras)
    
    logger.info("Predictions generated successfully.")

    # 4. Save and Submit
    pred_dataf = pd.DataFrame(preds, index=live_df.index, columns=["prediction"])
    
    Path(outputs_dir).mkdir(exist_ok=True)
    save_path = Path(outputs_dir) / "predictions.csv"
    pred_dataf.to_csv(save_path)
    logger.info(f"Predictions saved to {save_path}")

    # 5. Numerblox Submitter
    if auto_submit:
        logger.info(f"Auto-submitting for model: {model_name}")
        # The API credentials will be picked from the environment by default NumerAPI
        # or can be explicitly passed using numerblox.misc.Key
        import os
        pub_id = os.environ.get("NUMERAI_PUBLIC_ID")
        sec_key = os.environ.get("NUMERAI_SECRET_KEY")
        
        if not pub_id or not sec_key:
            logger.error("Auto Submit active, but NUMERAI_PUBLIC_ID or NUMERAI_SECRET_KEY not found in environment!")
        else:
            key = Key(pub_id=pub_id, secret_key=sec_key)
            submitter = NumeraiClassicSubmitter(directory_path=str(Path(outputs_dir)), key=key)
            submitter.full_submission(dataf=pred_dataf,
                                      cols="prediction",
                                      file_name="submission.csv",
                                      model_name=model_name)
            logger.info("Submission completed.")

    return pred_dataf


def main():
    parser = argparse.ArgumentParser(description="NumerBlox Live Inference")
    parser.add_argument("--model-name", required=True, help="Numerai model name")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--version", default="5.0")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    predict_live(
        model_name=args.model_name,
        data_dir=args.data_dir,
        version=args.version,
        auto_submit=args.submit,
        outputs_dir=args.outputs_dir,
    )


if __name__ == "__main__":
    main()
