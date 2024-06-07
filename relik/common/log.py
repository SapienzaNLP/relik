import logging
import os
import sys
import threading
from logging.config import dictConfig
from typing import Any, Dict, Optional

from art import text2art, tprint
from colorama import Fore, Style, init
from rich import get_console
from termcolor import colored, cprint


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

_default_log_level = logging.WARNING

# fancy logger
_console = get_console()


class ColorfulFormatter(logging.Formatter):
    """
    Formatter to add coloring to log messages by log type
    """

    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.CYAN,
        # "INFO": Fore.GREEN,
    }

    def format(self, record):
        record.rank = int(os.getenv("LOCAL_RANK", "0"))
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET


DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
        },
        "colorful": {
            "()": ColorfulFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] [RANK:%(rank)d] %(message)s",
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "filters": [],
            "stream": sys.stdout,
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "colorful",
            "filters": [],
            "stream": sys.stdout,
        },
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {
        "relik": {
            "handlers": ["color_console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}


def configure_logging(**kwargs):
    """Configure with default logging"""
    init()  # Initialize colorama
    # merge DEFAULT_LOGGING_CONFIG with kwargs
    logger_config = DEFAULT_LOGGING_CONFIG
    if kwargs:
        logger_config.update(kwargs)
    dictConfig(logger_config)


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_default_log_level)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def set_log_level(level: int, logger: logging.Logger = None) -> None:
    """
    Set the log level.
    Args:
        level (:obj:`int`):
            Logging level.
        logger (:obj:`logging.Logger`):
            Logger to set the log level.
    """
    if not logger:
        _configure_library_root_logger()
        logger = _get_library_root_logger()
    logger.setLevel(level)


def get_logger(
    name: Optional[str] = None,
    level: Optional[int] = None,
    formatter: Optional[str] = None,
    **kwargs,
) -> logging.Logger:
    """
    Return a logger with the specified name.
    """

    configure_logging(**kwargs)

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()

    if level is not None:
        set_log_level(level)

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    _default_handler.setFormatter(formatter)

    return logging.getLogger(name)


def get_console_logger():
    return _console


def print_relik_text_art(
    text: str = "relik", font: str = "larry3d", color: str = "magenta", **kwargs
):
    # tprint(text, font=font, **kwargs)
    art = text2art(text, font=font, **kwargs)#.rstrip()
    # art += "\n\n           Retrieve, Read, and Link"
    # art += "\nA fast and lightweight Information Extraction framework"
    cprint(art, color, attrs=["bold"])
