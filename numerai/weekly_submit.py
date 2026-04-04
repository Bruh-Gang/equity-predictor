"""
Numerai Weekly Auto-Submitter
------------------------------
Standalone script that runs predict.py with --submit for the weekly round.
No changes to predict.py or any other pipeline code.
Reads API credentials from environment variables set by auto_train.py.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Credentials (same as auto_train.py)
os.environ["NUMERAI_PUBLIC_ID"] = "GR6CSQYNJ5LGLFEOGI6ACBXXKZ4VLZ7Q"
os.environ["NUMERAI_SECRET_KEY"] = "ZXJPQ4GI42ACW56X5JOSKXFABLZCXRD7FBC5WXLHZ6SVMFUGAP2CBNQKUJJKCT4Y"

# Paths
SCRIPT_DIR = Path(__file__).parent
PREDICT_SCRIPT = SCRIPT_DIR / "predict.py"
MODEL_NAME = "nullsignal21345678"
DATA_DIR = str(SCRIPT_DIR / "data")
OUTPUTS_DIR = str(SCRIPT_DIR / "outputs")
LOG_FILE = SCRIPT_DIR / "outputs" / "weekly_submit.log"

def run_prediction():
    logger.info(f"=== Numerai Weekly Submission — {datetime.utcnow().isoformat()} UTC ===")

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--model-name", MODEL_NAME,
        "--data-dir", DATA_DIR,
        "--version", "5.0",
        "--outputs-dir", OUTPUTS_DIR,
        "--submit",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    if result.returncode == 0:
        logger.info("✓ Weekly submission completed successfully.")
    else:
        logger.error(f"✗ Submission failed with return code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_prediction()
