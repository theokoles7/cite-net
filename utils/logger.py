"""Logging configuration and initialization."""

from logging            import getLogger, Formatter, Logger, StreamHandler
from logging.handlers   import RotatingFileHandler
from os                 import makedirs
from sys                import stdout

from utils              import ARGS

# Ensure that logging path exists
makedirs(f"{ARGS.logging_path}", exist_ok=True)

# Initialize logger
LOGGER:             Logger =                getLogger('cite-net')

# Set logging level
LOGGER.setLevel(ARGS.logging_level)

# Define console handler
stdout_handler:     StreamHandler =         StreamHandler(stdout)
stdout_handler.setFormatter(Formatter("%(name)s | %(levelname)s | %(message)s"))
LOGGER.addHandler(stdout_handler)

# Define file handler
file_handler:       RotatingFileHandler =   RotatingFileHandler(f"{ARGS.logging_path}/{ARGS.cmd}.log", maxBytes = 1048576, backupCount = 3)
file_handler.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
LOGGER.addHandler(file_handler)