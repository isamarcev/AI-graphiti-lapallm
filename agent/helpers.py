"""
Helper functions для agent nodes.
Provides utility functions for ReAct reasoning, conflict detection, and message lookups.
"""

import logging
from typing import Optional
from db.neo4j_helpers import get_message_store

logger = logging.getLogger(__name__)


async def get_message_uid_by_episode(episode_name: str) -> Optional[str]:
    """
    Wrapper для Neo4j lookup - знаходить message UID по episode name.
    
    Args:
        episode_name: Name of the Graphiti episode
    
    Returns:
        Message UID or None if not found
    """
    try:
        store = await get_message_store()
        return await store.get_message_uid_by_episode(episode_name)
    except Exception as e:
        logger.error(f"Error in get_message_uid_by_episode: {e}")
        return None

