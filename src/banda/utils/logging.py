#  Copyright (c) 2025 by Karn Watcharasupat and contributors. All rights reserved.
#  This project is dual-licensed:
#  1. GNU Affero General Public License v3.0 (AGPLv3) for academic and non-commercial research use.
#     For details, see https://www.gnu.org/licenses/agpl-3.0.en.html
#  2. Commercial License for all other uses. Contact kwatcharasupat [at] ieee.org for commercial licensing.
#

import logging
import structlog
from typing import List, Union
import os
import hydra
import os
import hydra

def configure_logging(
    log_level: Union[int, str] = logging.INFO,
    log_format: str = "console",  # "console" or "json"
    log_file: str = None,
    loggers_to_silence: List[str] = None,
):
    """
    Configures structlog for structured logging.

    Args:
        log_level (Union[int, str]): The minimum logging level to capture.
        log_format (str): The format for log output ("console" or "json").
        log_file (str, optional): Path to a file to write logs to. Defaults to None.
        loggers_to_silence (List[str], optional): List of logger names to set to WARNING level.
                                                  Useful for silencing verbose libraries.
    """
    if loggers_to_silence is None:
        loggers_to_silence = ["matplotlib", "PIL", "h5py", "numba", "asyncio", "urllib3"]

    # Silence verbose loggers
    for logger_name in loggers_to_silence:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Configure structlog processors
    processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            }
        ),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:  # console format
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        
        cache_logger_on_first_use=True,
        
    )

    # Configure standard logging
    handlers = [logging.StreamHandler()]
    if log_file:
        # Resolve the log file path and create directories if they don't exist
        resolved_log_file = hydra.utils.to_absolute_path(log_file)
        log_dir = os.path.dirname(resolved_log_file)
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(resolved_log_file))

    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG,
        handlers=handlers,
    )

    # Set up a default logger for direct use
    logger = structlog.get_logger(__name__)
    logger.info("Logging configured successfully.", log_level=logging.DEBUG, log_format=log_format, log_file=log_file)
