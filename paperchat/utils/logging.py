"""Logging utilities for PaperChat components."""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logger(
    logger_name: str,
    log_file: str | None = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    propagate: bool = False,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        logger_name: Name of the logger
        log_file: Path to log file. If None, no file logging is configured.
        console_level: Logging level for console output
        file_level: Logging level for file output
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = propagate

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(file_level)
        file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_component_logger(component_name: str) -> logging.Logger:
    """Get a logger for a specific component with standard configuration.

    This creates a logger with both console and file output.

    Args:
        component_name: Name of the component (e.g., 'vector_store', 'embeddings')

    Returns:
        Configured logger for the component
    """
    log_dir = Path.home() / ".paperchat" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{component_name}.log"

    return setup_logger(
        logger_name=component_name,
        log_file=str(log_file),
        console_level=logging.WARNING,
        file_level=logging.DEBUG,
        propagate=False,
    )
