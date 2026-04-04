"""
Numerai Submission Utilities
-----------------------------
Handles prediction formatting, validation, and API submission.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_predictions(
    live_df: pd.DataFrame,
    predictions: np.ndarray,
    id_col: str = "id",
    pred_col: str = "prediction",
    clip: bool = True,
) -> pd.DataFrame:
    """
    Format predictions for Numerai submission.

    Args:
        live_df:     Live data with 'id' column.
        predictions: Array of predictions in [0, 1].
        clip:        Clip predictions to (0.01, 0.99) — avoids edge behavior.

    Returns:
        DataFrame with 'id' and 'prediction' columns.
    """
    if len(predictions) != len(live_df):
        raise ValueError(
            f"Prediction length {len(predictions)} != live data length {len(live_df)}"
        )

    submission = pd.DataFrame({
        id_col: live_df.index if id_col == "id" and id_col not in live_df.columns
                else live_df[id_col],
        pred_col: predictions,
    })

    # Rank-normalize to uniform [0,1]
    from scipy.stats import rankdata
    submission[pred_col] = rankdata(submission[pred_col]) / len(submission)

    if clip:
        submission[pred_col] = submission[pred_col].clip(0.01, 0.99)

    # Validate
    assert submission[pred_col].between(0, 1).all(), "Predictions out of [0,1] range!"
    assert not submission[pred_col].isna().any(), "NaN predictions detected!"
    assert len(submission) == len(live_df), "Submission length mismatch!"

    logger.info(f"Predictions: mean={submission[pred_col].mean():.4f}, "
                f"std={submission[pred_col].std():.4f}, "
                f"min={submission[pred_col].min():.4f}, "
                f"max={submission[pred_col].max():.4f}")
    return submission


def save_predictions(
    submission_df: pd.DataFrame,
    path: str = "outputs/predictions.csv",
) -> str:
    """Save predictions CSV to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(path, index=False)
    logger.info(f"Saved predictions to {path}")
    return path


def upload_predictions(
    submission_df: pd.DataFrame,
    model_name: str,
    public_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    tournament: int = 8,  # 8 = main tournament
) -> None:
    """
    Upload predictions to Numerai via the API.

    Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY environment variables,
    or pass them directly.

    Args:
        model_name:  Your model's name on Numerai (must match exactly).
        tournament:  8 for main tournament; see NumerAPI docs for others.
    """
    try:
        import numerapi
    except ImportError:
        raise ImportError("Run: pip install numerapi")

    pub = public_id or os.environ.get("NUMERAI_PUBLIC_ID", "")
    sec = secret_key or os.environ.get("NUMERAI_SECRET_KEY", "")

    if not pub or not sec:
        raise ValueError(
            "Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY environment variables "
            "or pass them to upload_predictions()."
        )

    napi = numerapi.NumerAPI(pub, sec)

    # Get model ID
    models = napi.get_models()
    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )
    model_id = models[model_name]
    logger.info(f"Found model '{model_name}' (ID: {model_id})")

    # Save to temp file and upload
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        submission_df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        submission_id = napi.upload_predictions(tmp_path, model_id=model_id)
        logger.info(f"Uploaded successfully! Submission ID: {submission_id}")
    finally:
        os.unlink(tmp_path)


def check_round_open() -> dict:
    """Check if the current Numerai round is open for submissions."""
    try:
        import numerapi
        napi = numerapi.NumerAPI()
        current_round = napi.get_current_round()
        return current_round
    except Exception as e:
        logger.warning(f"Could not check round status: {e}")
        return {}
