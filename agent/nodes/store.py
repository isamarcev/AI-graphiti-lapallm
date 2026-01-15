"""
Node: Store Knowledge
Saves teaching message directly to Graphiti (single LLM call for extraction).
Simplified version: Graphiti handles fact extraction internally.
"""

import logging
from typing import Dict, Any

from agent.state import AgentState
from clients.qdrant_client import QdrantClient
from clients.hosted_embedder import HostedQwenEmbedder
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="store_knowledge")
async def store_knowledge_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== Store Knowledge Node ===")

    message_uid = state["message_uid"]
    
    # Support for decomposed memory_updates from orchestrator
    memory_updates = state.get("memory_updates", [])
    if memory_updates:
        # Store all memory updates
        message_text = "\n".join(memory_updates)
        logger.info(f"Storing {len(memory_updates)} memory update(s)")
    else:
        message_text = state["message_text"]
    
    vector = state.get("message_embedding")
    if not vector:
        logger.warning("message_embedding missing; generating embedding in store node")
        embedder = HostedQwenEmbedder()
        vector = await embedder.create(message_text)

    # Save to Qdrant
    qdrant = await QdrantClient().initialize()
    try:
        # Use a new point id per insert to avoid overwriting if the caller reuses message_uid
        await qdrant.insert_record(
            record_id=None,
            vector=vector,
            fact=message_text,
            message_id=message_uid,
            is_relevant=True,
        )
        logger.info("Stored message in Qdrant", extra={"message_id": message_uid})
    finally:
        await qdrant.close()

    # Return stored count for learn response generation
    stored_count = len(memory_updates) if memory_updates else 1
    return {
        "message_embedding": vector,
        "stored_memories_count": stored_count
    }