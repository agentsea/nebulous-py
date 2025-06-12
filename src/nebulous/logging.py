import os
import sys

from loguru import logger

# --- Loguru Configuration ---
logger.remove()  # Remove default handler

# Get log level from environment variable, default to INFO, convert to uppercase
log_level_env = os.environ.get("PYTHON_LOG", "INFO").upper()

# Define valid log levels (uppercase)
valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

# Check if the provided level is valid
if log_level_env not in valid_levels:
    # Use print to stderr here as logger might not be fully configured yet
    print(
        f"Warning: Invalid PYTHON_LOG level '{log_level_env}'. "
        f"Valid levels are: {valid_levels}. Defaulting to 'INFO'.",
        file=sys.stderr,
    )
    log_level = "INFO"
else:
    log_level = log_level_env

# Add new handler with the configured level
logger.add(sys.stderr, level=log_level)

# --- End Loguru Configuration ---

# Export the configured logger
__all__ = ["logger"]
