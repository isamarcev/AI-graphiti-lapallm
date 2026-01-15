"""
Node: Resolve Conflicts
Marks conflicting records in Qdrant as not relevant (is_relevant=False) by message_id.
"""

import logging
from typing import Any, Dict, List, Tuple

from langsmith import traceable
from qdrant_client.http import models as qmodels

from agent.state import AgentState
from clients.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


async def _mark_message_irrelevant(qdrant: QdrantClient, message_id: str):
    await qdrant.set_relevance_by_message_id(message_id=message_id, is_relevant=False)


@traceable(name="resolve_conflicts")
async def resolve_conflicts_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== Resolve Conflicts Node ===")
    conflicts: List[Tuple[str, str]] = state.get("conflicts") or []

    if not conflicts:
        logger.info("No conflicts to resolve; passing through")
        return {"conflicts": []}

    qdrant = await QdrantClient().initialize()
    try:
        for msg_id, _fact in conflicts:
            try:
                await _mark_message_irrelevant(qdrant, msg_id)
                logger.info(f"Marked message_id={msg_id} as not relevant in Qdrant")
            except Exception as e:
                logger.error(f"Failed to mark message_id={msg_id}: {e}")
    finally:
        await qdrant.close()

    # Return empty conflicts after resolution
    return {}
