import logging
from typing import Any, Dict

from agent.state import AgentState
from clients.hosted_embedder import HostedQwenEmbedder
from clients.qdrant_client import QdrantClient
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="index_raw")
async def index_raw_node(state: AgentState) -> Dict[str, Any]:
    """
    Store raw memory updates directly in vector store without decomposition.
    
    Takes items from state["memory_updates"] and stores them as-is,
    also vectorizing the original raw message text.
    """
    
    logger.info("=== Index Raw Node ===")
    
    memory_updates = state.get("memory_updates") or []
    
    if not memory_updates:
        logger.warning("index_raw_node: no memory_updates to store")
        return {}
    
    embedder = HostedQwenEmbedder()
    qdrant = await QdrantClient().initialize()
    
    stored_count = 0
    indexed_facts = []
    
    try:
        message_uid = state.get("message_uid", "")
        
        for update_text in memory_updates:
            if not update_text:
                logger.warning("Skipping empty memory update")
                continue
            
            update_vector = await embedder.create(update_text)
            
            await qdrant.insert_record(
                vector=update_vector,
                fact=update_text,
                message_id=message_uid,
                is_relevant=True,
            )
            
            indexed_facts.append({
                "fact": update_text,
                "description": "",
                "examples": ""
            })
            
            stored_count += 1
            logger.info(
                f"Stored raw memory update {stored_count} in Qdrant",
                extra={
                    "message_id": message_uid,
                    "update_text": update_text[:50],
                },
            )
    finally:
        await qdrant.close()
    
    logger.info(f"Stored {stored_count} raw memory updates total")
    return {"indexed_facts": indexed_facts}
