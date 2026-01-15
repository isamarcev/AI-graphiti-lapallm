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
    message_text = state["message_text"]
    vector = state.get("message_embedding")
    if not vector:
        logger.warning("message_embedding missing; generating embedding in store node")
        embedder = HostedQwenEmbedder()
        vector = await embedder.create(message_text)

    # Save to Qdrant
    qdrant = await QdrantClient().initialize()
    try:
        await qdrant.insert_record(
            record_id=message_uid,
            vector=vector,
            fact=message_text,
            message_id=message_uid,
            is_relevant=True,
        )
        logger.info("Stored message in Qdrant", extra={"message_id": message_uid})
    finally:
        await qdrant.close()

    return {"message_embedding": vector}