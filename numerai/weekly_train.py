"""
Numerai Weekly Retrainer
--------------------------
Run this once and leave it running. Every Monday night it:
  1. Re-downloads the latest weekly training data from Numerai
  2. Retrains the full MetaPipeline via train.py
  3. Immediately predicts + submits for that week's live round

This keeps your model fresh on new data every week automatically.

Just run:  python weekly_train.py

No external scheduler needed. No changes to train.py or predict.py.
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Credentials (same as auto_train.py) ──────────────────────────────────────
os.environ["NUMERAI_PUBLIC_ID"]  = "GR6CSQYNJ5LGLFEOGI6ACBXXKZ4VLZ7Q"
os.environ["NUMERAI_SECRET_KEY"] = "ZXJPQ4GI42ACW56X5JOSKXFABLZCXRD7FBC5WXLHZ6SVMFUGAP2CBNQKUJJKCT4Y"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
TRAIN_PY    = SCRIPT_DIR / "train.py"
PREDICT_PY  = SCRIPT_DIR / "predict.py"
DATA_DIR    = SCRIPT_DIR / "data"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
LOG_FILE    = OUTPUTS_DIR / "weekly_train.log"
MODEL_NAME  = "nullsignal21345678"
VERSION     = "5.0"
N_SPLITS    = 5
NEUTRALIZATION = 0.5

# ── Retrain window: Monday at 22:00 UTC (before Tuesday live drop) ────────────
RETRAIN_DAY_UTC  = 0    # Monday
RETRAIN_HOUR_UTC = 22

# ── Logging ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def seconds_until_next_retrain(now: datetime) -> float:
    """
    Calculate seconds until the next Monday at 22:00 UTC.
    """
    candidate = now.replace(hour=RETRAIN_HOUR_UTC, minute=0, second=0, microsecond=0)

    # Advance to the next Monday
    days_ahead = (RETRAIN_DAY_UTC - now.weekday()) % 7
    candidate += timedelta(days=days_ahead)

    # If that time already passed this week, jump to next week
    if now >= candidate:
        candidate += timedelta(weeks=1)

    return (candidate - now).total_seconds()


def run(cmd: list[str]) -> int:
    """Stream a subprocess and return its exit code."""
    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        logger.info(line.rstrip())
    proc.wait()
    return proc.returncode


def do_weekly_cycle():
    """Full retrain → predict → submit cycle."""
    now = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info(f"WEEKLY RETRAIN START — {now.strftime('%A %Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 70)

    # Step 1: Retrain
    logger.info("STEP 1 — Downloading latest data and retraining pipeline...")
    ret_train = run([
        sys.executable, str(TRAIN_PY),
        "--data-dir",       str(DATA_DIR),
        "--version",        VERSION,
        "--n-splits",       str(N_SPLITS),
        "--neutralization", str(NEUTRALIZATION),
    ])

    if ret_train != 0:
        logger.error(f"✗ train.py failed (code {ret_train}). Skipping predict step this week.")
        return

    logger.info("✓ Retraining complete.")

    # Step 2: Predict + Submit with the fresh model
    logger.info("STEP 2 — Predicting and submitting with freshly trained model...")
    ret_pred = run([
        sys.executable, str(PREDICT_PY),
        "--model-name",  MODEL_NAME,
        "--data-dir",    str(DATA_DIR),
        "--version",     VERSION,
        "--outputs-dir", str(OUTPUTS_DIR),
        "--submit",
    ])

    if ret_pred == 0:
        logger.info("✓ Weekly retrain + submission complete.")
    else:
        logger.error(f"✗ predict.py failed (code {ret_pred}). Check {LOG_FILE}.")


def main():
    logger.info("=" * 70)
    logger.info("Numerai Weekly Retrainer started — retrains every Monday night.")
    logger.info("Leave this process running. Press Ctrl+C to stop.")
    logger.info("=" * 70)

    retrained_this_week: set[str] = set()   # ISO week string e.g. "2026-W15"

    while True:
        now      = datetime.now(timezone.utc)
        iso_week = now.strftime("%G-W%V")   # e.g. "2026-W15"

        # ── It's Monday night and we haven't retrained this week yet ──────────
        if (now.weekday() == RETRAIN_DAY_UTC
                and now.hour >= RETRAIN_HOUR_UTC
                and iso_week not in retrained_this_week):

            do_weekly_cycle()
            retrained_this_week.add(iso_week)

        # ── Sleep until next Monday 22:00 UTC ─────────────────────────────────
        wait_secs = seconds_until_next_retrain(now)
        wake_at   = now + timedelta(seconds=wait_secs)
        logger.info(
            f"Next retrain: {wake_at.strftime('%A %Y-%m-%d %H:%M UTC')} "
            f"({wait_secs / 3600:.1f} hrs away). Sleeping..."
        )
        time.sleep(wait_secs)


if __name__ == "__main__":
    main()
