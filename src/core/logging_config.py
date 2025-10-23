import logging
import logging.config
from typing import Optional


def configure_logging(level: int = logging.INFO, loggers: Optional[dict] = None) -> None:
    """Configure structured logging for the application."""

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
        "loggers": loggers or {},
    }

    logging.config.dictConfig(logging_config)


__all__ = ["configure_logging"]
