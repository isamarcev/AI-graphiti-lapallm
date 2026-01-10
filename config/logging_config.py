"""
Logging configuration for the application.

Suppresses non-critical warnings from Neo4j and Graphiti:
- EquivalentSchemaRuleAlreadyExists errors (indexes already exist)
- Missing property warnings (properties don't exist yet in empty graph)
"""

import logging
import warnings


class Neo4jSchemaWarningFilter(logging.Filter):
    """Filter out non-critical Neo4j schema warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to suppress specific Neo4j warnings.

        Returns False to suppress the record, True to allow it.
        """
        message = record.getMessage()

        # Suppress "EquivalentSchemaRuleAlreadyExists" errors
        if "EquivalentSchemaRuleAlreadyExists" in message:
            return False

        # Suppress "Error executing Neo4j query" for index creation
        if "Error executing Neo4j query" in message and "CREATE INDEX" in message:
            return False

        # Allow all other messages
        return True


class Neo4jNotificationFilter(logging.Filter):
    """Filter out Neo4j notifications about missing properties."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to suppress Neo4j property warnings.

        Returns False to suppress the record, True to allow it.
        """
        message = record.getMessage()

        # Suppress "Received notification from DBMS server" for missing properties
        if "Received notification from DBMS server" in message:
            # Allow errors, suppress warnings about missing properties
            if "property key does not exist" in message:
                return False
            if "does not exist. Verify that the spelling is correct" in message:
                return False

        # Allow all other messages
        return True


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

    # Configure graphiti_core logger to WARNING (suppress INFO messages)
    graphiti_logger = logging.getLogger('graphiti_core')
    graphiti_logger.setLevel(logging.WARNING)
    graphiti_logger.addFilter(Neo4jSchemaWarningFilter())
    graphiti_logger.addFilter(Neo4jNotificationFilter())

    # Configure neo4j logger to WARNING
    neo4j_logger = logging.getLogger('neo4j')
    neo4j_logger.setLevel(logging.WARNING)
    neo4j_logger.addFilter(Neo4jSchemaWarningFilter())
    neo4j_logger.addFilter(Neo4jNotificationFilter())

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