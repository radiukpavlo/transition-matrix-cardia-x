"""Structured logging configuration for the tm-ecg pipeline."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbosity: int = 1) -> None:
    """Configure structured logging for the pipeline.

    Parameters
    ----------
    verbosity:
        0 = WARNING only, 1 = INFO (default), 2 = DEBUG.
    """
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.INFO
    )
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger("tm_ecg")
    root.setLevel(level)

    if not root.handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(formatter)
