import logging
import structlog
import sys

def configure_logging(log_level: str = "INFO"):
    """
    Configures structlog and standard Python logging.

    Args:
        log_level (str): The minimum logging level to capture (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
    """
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.dev.ConsoleRenderer()
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog for specific loggers if needed
    # For example, to set a different level for a specific module:
    # logging.getLogger("banda.models").setLevel(logging.DEBUG)
