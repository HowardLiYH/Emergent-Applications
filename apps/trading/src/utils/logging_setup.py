"""
Logging setup for experiment tracking.
"""
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging with file and console handlers.

    Usage:
        logger = setup_logging("experiment")
        logger.info("Starting experiment")
    """
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    log_file = f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")

    return logger
