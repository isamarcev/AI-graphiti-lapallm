"""
Node 5: Store Knowledge
Saves teaching message to Graphiti (knowledge graph) and Neo4j (references).
"""

import logging
from typing import Dict, Any
from datetime import datetime

from agent.state import AgentState
from clients.graphiti_client import get_graphiti_client
from db.neo4j_helpers import get_message_store
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="store_knowledge")
async def store_knowledge_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: Store knowledge in both Graphiti and Neo4j.

    Workflow:
    1. Save episode to Graphiti (knowledge graph with entities/relations)
    2. Save raw message to Neo4j (for UUID references)
    3. Link them via episode_name

    Args:
        state: Current agent state

    Returns:
        State update with response confirmation
    """
    logger.info("=== Store Knowledge Node ===")
    logger.info(f"Storing message {state['message_uid']}: {state['message_text'][:80]}...")

    graphiti = await get_graphiti_client()
    message_store = await get_message_store()

    try:
        # 1. Add episode to Graphiti
        episode_name = f"teach_{state['message_uid']}"

        # Format episode as a proper conversation
        # This is critical for Graphiti's LLM to parse correctly
        user_message = state["message_text"]
        assistant_message = state.get("confirmation_text") or "Дякую за інформацію. Я її зберіг."

        episode_body = f"""User: {user_message}
Assistant: {assistant_message}"""

        logger.info(f"Adding episode to Graphiti: {episode_name}")
        logger.debug(f"Episode body:\n{episode_body}")

        episode = await graphiti.add_episode(
            episode_body=episode_body,
            episode_name=episode_name,
            source_description=f"user:{state['user_id']}, uid:{state['message_uid']}",
            reference_time=state["timestamp"]
        )

        episode_id = episode.episode_id if hasattr(episode, 'episode_id') else episode_name
        logger.info(f"Episode saved with ID: {episode_id}")

        # 2. Save message to Neo4j
        logger.info(f"Saving message to Neo4j")

        await message_store.save_message(
            uid=state["message_uid"],
            text=state["message_text"],
            episode_name=episode_name,
            user_id=state["user_id"],
            timestamp=state["timestamp"]
        )

        logger.info(f"Message saved to Neo4j and linked to episode")

        # 3. Count extracted facts
        fact_count = len(state.get("extracted_facts", []))

        response = f"✓ Навчання збережено.\n\nВивчено {fact_count} фактів з повідомлення {state['message_uid']}."

        return {
            "response": response,
            "references": [state["message_uid"]],
            "confidence": 1.0
        }

    except Exception as e:
        logger.error(f"Error storing knowledge: {e}", exc_info=True)
        return {
            "response": f"Помилка при збереженні знань: {e}",
            "references": [],
            "confidence": 0.0
        }
