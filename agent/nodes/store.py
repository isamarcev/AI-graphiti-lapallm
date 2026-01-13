"""
Node: Store Knowledge
Saves teaching message directly to Graphiti (single LLM call for extraction).
Simplified version: Graphiti handles fact extraction internally.
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
    Store knowledge in both Graphiti and Neo4j.

    Simplified workflow (single LLM call):
    1. Save episode to Graphiti (extracts facts internally)
    2. Save raw message to Neo4j (for message_uid references)
    3. Log extracted entities/relations

    Args:
        state: Current agent state with message_text

    Returns:
        State update with response confirmation and extracted entity count
    """
    logger.info("=== Store Knowledge Node (Simplified) ===")
    logger.info(f"Storing message {state['message_uid']}: {state['message_text'][:80]}...")

    graphiti = await get_graphiti_client()
    message_store = await get_message_store()

    try:
        # 1. Add episode to Graphiti (single LLM call, Graphiti extracts facts)
        episode_name = f"teach_{state['message_uid']}"

        # Format as proper conversation with user/assistant alternation
        # CRITICAL: LiteLLM requires proper role alternation
        user_message = state["message_text"]
        assistant_message = "Дякую за інформацію. Я її зберіг."

        episode_body = f"""User: {user_message}
Assistant: {assistant_message}"""

        logger.info(f"Adding episode to Graphiti: {episode_name}")
        logger.info(f"Episode body:\n{episode_body}")

        episode = await graphiti.add_episode(
            episode_body=episode_body,
            episode_name=episode_name,
            source_description=f"user:{state['user_id']}, uid:{state['message_uid']}",
            reference_time=state["timestamp"]
        )

        # Extract episode result info
        episode_id = episode.uuid if hasattr(episode, 'uuid') else episode_name
        logger.info(f"✓ Episode saved with ID: {episode_id}")

        # Log extracted entities and relations (if available)
        if hasattr(episode, 'nodes') and episode.nodes:
            entity_names = [node.name for node in episode.nodes if hasattr(node, 'name')]
            logger.info(f"📦 Extracted {len(entity_names)} entities: {entity_names}")

        if hasattr(episode, 'edges') and episode.edges:
            relations = [
                f"{edge.source_node_name} -> {edge.name} -> {edge.target_node_name}"
                for edge in episode.edges
                if hasattr(edge, 'name') and hasattr(edge, 'source_node_name') and hasattr(edge, 'target_node_name')
            ]
            logger.info(f"🔗 Extracted {len(relations)} relations: {relations}")

        # 2. Save message to Neo4j for references
        logger.info(f"Saving message to Neo4j")

        await message_store.save_message(
            uid=state["message_uid"],
            text=state["message_text"],
            episode_name=episode_name,
            user_id=state["user_id"],
            timestamp=state["timestamp"]
        )

        logger.info(f"✓ Message saved to Neo4j and linked to episode")

        # 3. Prepare response
        entity_count = len(episode.nodes) if hasattr(episode, 'nodes') else 0
        response = f"✓ Навчання збережено.\n\nВивчено {entity_count} сутностей з повідомлення {state['message_uid']}."

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
