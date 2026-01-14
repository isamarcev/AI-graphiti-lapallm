"""
Neo4j helpers для збереження та пошуку message references.
Зберігаємо Message nodes пов'язані з Episode nodes.
"""

import logging
from typing import Optional
from neo4j import AsyncGraphDatabase
from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jMessageStore:
    """
    Helper для роботи з Message nodes в Neo4j.
    
    Зберігає messages як окремі nodes пов'язані з Episode nodes,
    що дозволяє:
    - Знаходити source message по episode_id
    - Надавати references у відповідях
    - Trackати conversation history
    """
    
    def __init__(self):
        """Initialize Neo4j driver with settings from config"""
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        logger.info(f"Neo4jMessageStore initialized with URI: {settings.neo4j_uri}")
    
    async def save_message(
        self,
        uid: str,
        text: str,
        episode_name: str,
        user_id: str,
        timestamp
    ) -> None:
        """
        Зберігає message як окремий node з зв'язком до episode.
        
        Structure:
        (Message {uid, text, timestamp, user_id})-[:GENERATED_EPISODE]->(Episode {name})
        
        Args:
            uid: Unique message identifier (UUID)
            text: Message content
            episode_name: Name of the Graphiti episode
            user_id: User identifier
            timestamp: Message timestamp (datetime object)
        """
        query = """
        MERGE (e:Episode {name: $episode_name})
        CREATE (m:Message {
            uid: $uid,
            text: $text,
            timestamp: datetime($timestamp),
            user_id: $user_id
        })
        CREATE (m)-[:GENERATED_EPISODE]->(e)
        RETURN m.uid as uid
        """
        
        try:
            async with self.driver.session(database=settings.neo4j_database) as session:
                result = await session.run(
                    query,
                    uid=uid,
                    text=text,
                    episode_name=episode_name,
                    user_id=user_id,
                    timestamp=timestamp.isoformat()
                )
                await result.single()
                logger.info(f"Message {uid} saved and linked to episode {episode_name}")
        except Exception as e:
            logger.error(f"Error saving message {uid}: {e}", exc_info=True)
            raise
    
    async def get_message_uid_by_episode(self, episode_name: str) -> Optional[str]:
        """
        Знаходить message UID по episode name.
        
        Args:
            episode_name: Name of the Graphiti episode
        
        Returns:
            Message UID or None if not found
        """
        query = """
        MATCH (m:Message)-[:GENERATED_EPISODE]->(e:Episode {name: $episode_name})
        RETURN m.uid as uid
        LIMIT 1
        """
        
        try:
            async with self.driver.session(database=settings.neo4j_database) as session:
                result = await session.run(query, episode_name=episode_name)
                record = await result.single()
                
                if record:
                    uid = record["uid"]
                    logger.debug(f"Found message UID {uid} for episode {episode_name}")
                    return uid
                else:
                    logger.debug(f"No message found for episode {episode_name}")
                    return None
        except Exception as e:
            logger.error(f"Error retrieving message UID for episode {episode_name}: {e}")
            return None

    
    async def close(self):
        """Close Neo4j driver connection"""
        await self.driver.close()
        logger.info("Neo4jMessageStore connection closed")


# Global instance
_message_store: Optional[Neo4jMessageStore] = None


async def get_message_store() -> Neo4jMessageStore:
    """
    Get or create the global Neo4jMessageStore instance.
    
    Returns:
        Initialized Neo4jMessageStore
    """
    global _message_store
    if _message_store is None:
        _message_store = Neo4jMessageStore()
        logger.debug("Created new Neo4jMessageStore instance")
    return _message_store
