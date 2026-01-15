"""
Logging configuration for the application.

Suppresses non-critical warnings from Neo4j and Graphiti:
- EquivalentSchemaRuleAlreadyExists errors (indexes already exist)
- Missing property warnings (properties don't exist yet in empty graph)
"""

import logging
import warnings


def configure_logging(level: str = "INFO"):
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Set root logger level
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Keep our application loggers at INFO
    app_logger = logging.getLogger('agent')
    app_logger.setLevel(level)

    clients_logger = logging.getLogger('clients')
    clients_logger.setLevel(level)

    # Suppress Python warnings about unclosed resources (common in async code)
    warnings.filterwarnings('ignore', category=ResourceWarning)

    logging.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with configured filters.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)