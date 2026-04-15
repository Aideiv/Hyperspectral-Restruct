"""
Centralized logging configuration for Hyperspectral-Restruct.

Provides consistent logging across all modules with configurable
log levels, formatting, and output destinations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure root logger for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file for file output
        format_string: Custom format string (uses default if None)
        console: Whether to output to console
        
    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger
    root_logger = logging.getLogger("hyperspectral")
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance with 'hyperspectral.' prefix
    """
    return logging.getLogger(f"hyperspectral.{name}")


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emojis to log levels."""
    
    LEVEL_EMOJIS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Add emoji prefix to log message."""
        emoji = self.LEVEL_EMOJIS.get(record.levelno, "")
        record.msg = f"{emoji} {record.msg}"
        return super().format(record)


def setup_emoji_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging with emoji prefixes for better console visibility."""
    formatter = EmojiFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    root_logger = logging.getLogger("hyperspectral")
    root_logger.setLevel(level)
    root_logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger
