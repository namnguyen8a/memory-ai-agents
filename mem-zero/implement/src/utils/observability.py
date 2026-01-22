"""Observability utilities: colored logger + timing decorator.

This module is intentionally simple and self-contained (stdlib only).
You can reuse it in production; for production you may want:
- JSON logging (structlog / python-json-logger)
- OpenTelemetry tracing + metrics
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ColorFormatter(logging.Formatter):
    """Simple colored formatter for console output."""

    COLORS = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logger(name: str = "RAG_Pipeline") -> logging.Logger:
    """Cấu hình logger chuẩn với màu sắc."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Xóa handler cũ nếu đã tồn tại (tránh duplicate log)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Default logger used across the demo.
logger = setup_logger()


def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure wall-clock time of a function and log it.

    Logs:
      [TIME] 'func_name': 0.1234s
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"[TIME] '{func.__name__}': {elapsed:.4f}s")

    return wrapper


