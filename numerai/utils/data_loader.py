"""
Numerai Data Loader with NumerBlox
----------------------------------
Handles downloading, caching, and loading Numerai tournament data 
efficiently into NumerFrames.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from numerblox.download import NumeraiClassicDownloader
from numerblox.numerframe import create_numerframe

logger = logging.getLogger(__name__)


def download_data(
    data_dir: str = "data",
    version: str = "5.0",
) -> Path:
    """
    Download Numerai dataset to local disk using NumerBlox.
    
    Args:
        data_dir: Directory to store downloaded files.
        version: Dataset version (default: 5.0).
        
    Returns:
        Path to the versioned data directory.
    """
    downloader = NumeraiClassicDownloader(data_dir)
    
    logger.info("Initializing download of training and validation data...")
    # NumerBlox will skip download if files exist and are complete
    downloader.download_training_data("train_val", version=version)
    
    data_path = Path(data_dir) / "train_val"
    return data_path


def load_numerframe_data(
    data_path: str,
    file_name: str = "train.parquet",
) -> pd.DataFrame:
    """
    Load data directly into a memory-efficient NumerFrame.
    
    Args:
        data_path: Path to the directory containing parquet file.
        file_name: Name of the parquet file.
        
    Returns:
        NumerFrame (Pandas DataFrame subclass with Numerai hooks)
    """
    full_path = Path(data_path) / file_name
    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {full_path}")
        
    logger.info(f"Loading {file_name} into NumerFrame...")
    
    # create_numerframe loads the parquet file efficiently
    df = create_numerframe(str(full_path))
    
    logger.info(f"Loaded {len(df)} rows. Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df
