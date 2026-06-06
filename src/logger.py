"""
logger.py
=========
Centralized logging configuration for Medical AI Platform.
Sets up colored console logging + rotating file logging.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
import colorlog

# ─────────────────────────────────────────────
# Create logs directory if it doesn't exist
# ─────────────────────────────────────────────
os.makedirs("./logs", exist_ok=True)


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name (str): Logger name (usually __name__ of the calling module)
        log_level (str): Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # ── Colored Console Handler ──────────────────────────────────────
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        }
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # ── Rotating File Handler ────────────────────────────────────────
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler(
        filename="./logs/medical_ai.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


# ── Module-level default logger ──────────────────────────────────────────────
logger = get_logger("medical_ai")
