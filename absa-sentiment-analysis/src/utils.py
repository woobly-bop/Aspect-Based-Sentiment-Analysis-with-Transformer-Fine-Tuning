"""
Utilities: seeding, logging, device detection for reproducibility and debugging.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Default log format
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the best available device (CUDA if available and requested, else CPU).

    Args:
        use_cuda: If True, prefer GPU when available.

    Returns:
        torch.device
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    name: str = "absa",
) -> logging.Logger:
    """
    Configure structured logging to console and optionally to a file.

    Args:
        log_file: If set, also write logs to this file.
        level: Logging level (default INFO).
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_experiment_name(model_name: str, seed: Optional[int] = None) -> str:
    """
    Generate a simple experiment name for checkpointing and metrics.

    Args:
        model_name: baseline, lstm, or bert.
        seed: Optional seed to include in name.

    Returns:
        String like "baseline_s42" or "bert".
    """
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed is not None:
        return f"{model_name}_s{seed}_{stamp}"
    return f"{model_name}_{stamp}"
