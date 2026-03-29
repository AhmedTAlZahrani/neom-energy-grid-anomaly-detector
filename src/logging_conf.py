"""Dict-based logging configuration for the anomaly detector.

Provides a console handler at DEBUG level and a file handler at WARNING
level writing to ``logs/detector.log``.
"""

import logging.config
from pathlib import Path


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "brief": {
            "format": "%(levelname)-8s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "brief",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "WARNING",
            "formatter": "detailed",
            "filename": "logs/detector.log",
            "mode": "a",
        },
    },
    "loggers": {
        "anomaly_detector": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}


def setup():
    """Apply the dict-based logging configuration."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
