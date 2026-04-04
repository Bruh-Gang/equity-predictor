"""
Numerai Daily Predictor & Submitter
-------------------------------------
Run this once and leave it running. It wakes up every day Tue–Sat when
Numerai releases live data, downloads the latest round, generates predictions
using the saved MetaPipeline, and auto-submits.

Just run:  python daily_predict.py

No external scheduler needed. No changes to predict.py or any other file.
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
PREDICT_PY  = SCRIPT_DIR / "predict.py"
DATA_DIR    = SCRIPT_DIR / "data"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
LOG_FILE    = OUTPUTS_DIR / "daily_predict.log"
MODEL_NAME  = "nullsignal21345678"
VERSION     = "5.0"

# ── Submission window: Numerai live data drops ~6–8 AM UTC Tue–Sat ───────────
SUBMIT_HOUR_UTC = 9    # Run at 9:00 AM UTC (safe margin after data drop)
LIVE_DAYS = {1, 2, 3, 4, 5}  # Mon=0 … Sat=5. Tue=1, Wed=2, Thu=3, Fri=4, Sat=5

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


def pipeline_exists() -> bool:
    return (OUTPUTS_DIR / "numerblox_meta_pipeline.pkl").exists()


def seconds_until_next_run(now: datetime) -> float:
    """
    Calculate seconds until the next valid submission window:
    9:00 AM UTC on the next Tue–Sat.
    """
    candidate = now.replace(hour=SUBMIT_HOUR_UTC, minute=0, second=0, microsecond=0)

    # If today's window already passed, start looking from tomorrow
    if now >= candidate:
        candidate += timedelta(days=1)

    # Advance until we land on a Tue–Sat
    while candidate.weekday() not in LIVE_DAYS:
        candidate += timedelta(days=1)

    return (candidate - now).total_seconds()


def run_predict() -> int:
    """Invoke predict.py and stream its output. Returns exit code."""
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--model-name",  MODEL_NAME,
        "--data-dir",    str(DATA_DIR),
        "--version",     VERSION,
        "--outputs-dir", str(OUTPUTS_DIR),
        "--submit",
    ]
    logger.info(f"Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        logger.info(line.rstrip())
    proc.wait()
    return proc.returncode


def main():
    logger.info("=" * 70)
    logger.info("Numerai Daily Predictor started — will run automatically Tue–Sat.")
    logger.info("Leave this process running. Press Ctrl+C to stop.")
    logger.info("=" * 70)

    submitted_today: set[str] = set()   # track dates already submitted

    while True:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        weekday  = now.weekday()   # Mon=0, Tue=1, …, Sat=5, Sun=6

        # ── It's a live day and we haven't submitted yet today ────────────────
        if weekday in LIVE_DAYS and now.hour >= SUBMIT_HOUR_UTC and date_str not in submitted_today:

            logger.info(f"Live day detected ({now.strftime('%A')}). Attempting submission...")

            if not pipeline_exists():
                logger.error(
                    "No trained pipeline found. Run weekly_train.py first, "
                    "then restart daily_predict.py."
                )
            else:
                ret = run_predict()
                if ret == 0:
                    submitted_today.add(date_str)
                    logger.info(f"✓ Submission complete for {date_str}.")
                else:
                    logger.error(f"✗ predict.py failed (code {ret}). Will retry in 30 minutes.")
                    time.sleep(30 * 60)
                    continue

        # ── Sleep until the next submission window ────────────────────────────
        wait_secs = seconds_until_next_run(now)
        wake_at   = now + timedelta(seconds=wait_secs)
        logger.info(
            f"Next submission window: {wake_at.strftime('%A %Y-%m-%d %H:%M UTC')} "
            f"({wait_secs / 3600:.1f} hrs away). Sleeping..."
        )
        time.sleep(wait_secs)


if __name__ == "__main__":
    main()
