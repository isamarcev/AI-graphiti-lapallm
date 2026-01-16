import logging
from typing import Any, Dict

from agent.state import AgentState
from clients.hosted_embedder import HostedQwenEmbedder
from clients.qdrant_client import QdrantClient
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="store_indexed_facts")
async def store_indexed_facts_node(state: AgentState) -> Dict[str, Any]:
    """
    Persist indexed facts to Qdrant.

    Expects state["indexed_facts"] as a list of dicts with keys:
      - fact: str
      - description: str (optional)
      - examples: list[str] (optional)

    Embeds each fact text and writes records using message_uid as message_id.
    """

    indexed_facts = state.get("indexed_facts") or []
    
    if not indexed_facts:
        logger.warning("store_indexed_fact_node: no indexed_facts to store")
        return {}

    embedder = HostedQwenEmbedder()
    qdrant = await QdrantClient().initialize()
    
    stored_count = 0
    last_vector = None
    
    try:
        for indexed in indexed_facts:
            fact_text = indexed.get("fact") or ""
            description = indexed.get("description")
            examples = indexed.get("examples")

            if not fact_text:
                logger.warning("Skipping indexed fact with empty fact field")
                continue

            vector = await embedder.create(description)
            last_vector = vector

            await qdrant.insert_record(
                vector=vector,
                fact=fact_text,
                message_id=state.get("message_uid", ""),
                is_relevant=True,
                payload={"description": description, "examples": examples},
            )
            
            stored_count += 1
            logger.info(
                f"Stored indexed fact {stored_count} in Qdrant",
                extra={
                    "message_id": state.get("message_uid"),
                    "fact": fact_text[:50],
                    "has_description": bool(description),
                    "examples_count": len(examples) if isinstance(examples, list) else 0,
                },
            )
    finally:
        await qdrant.close()

    logger.info(f"Stored {stored_count} indexed facts total")
    return {"message_embedding": last_vector or []}