from pathlib import Path
import random

import numpy as np

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
)


def create_directories() -> None:
    """
    Create the required project directories if they do not already exist.
    """

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)