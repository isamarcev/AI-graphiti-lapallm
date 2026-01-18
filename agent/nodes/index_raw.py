import logging
from typing import Any, Dict

from agent.state import AgentState
from clients.hosted_embedder import HostedQwenEmbedder
from clients.llm_client import get_llm_client
from clients.qdrant_client import QdrantClient
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="index_raw")
async def index_raw_node(state: AgentState) -> Dict[str, Any]:
    """
    Store raw memory updates directly in vector store without decomposition.
    
    Takes items from state["memory_updates"] and stores them as-is,
    also extracting brief facts from the original message text using LLM.
    """
    
    logger.info("=== Index Raw Node ===")
    
    memory_updates = state.get("memory_updates") or []
    
    if not memory_updates:
        logger.warning("index_raw_node: no memory_updates to store")
        return {}
    
    embedder = HostedQwenEmbedder()
    qdrant = await QdrantClient().initialize()
    llm_client = get_llm_client()
    
    stored_count = 0
    indexed_facts = []
    
    try:
        message_uid = state.get("message_uid", "")
        message_text = state.get("message_text", "")
        
        # Extract brief fact from message text using LLM (Ukrainian prompt)
        brief_fact = ""
        if message_text:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "Ти асистент для витягування фактів. Витягни ключовий факт з повідомлення користувача. Будь стислим і фактичним (максимум 1-2 речення)."
                    },
                    {
                        "role": "user",
                        "content": f"Витягни ключовий факт з цього повідомлення:\n\n{message_text}"
                    }
                ]
                brief_fact = await llm_client.generate_async(messages, temperature=0.3, max_tokens=100)
                logger.info(f"Extracted brief fact: {brief_fact[:100]}")
            except Exception as e:
                logger.error(f"Failed to extract brief fact: {e}")
                brief_fact = ""
        
        # Create vector from message_text
        message_vector = await embedder.create(message_text) if message_text else None
        
        for update_text in memory_updates:
            if not update_text:
                logger.warning("Skipping empty memory update")
                continue
            
            # Use message_text vector if available, otherwise create from update_text
            vector_to_store = message_vector if message_vector else await embedder.create(update_text)
            
            # Store fact (whole message_text) and brief_fact separately
            await qdrant.insert_record(
                vector=vector_to_store,
                fact=message_text if message_text else update_text,
                message_id=message_uid,
                is_relevant=True,
                brief_fact=brief_fact if brief_fact else None,
            )
            
            indexed_facts.append({
                "fact": message_text if message_text else update_text,
                "brief_fact": brief_fact
            })
            
            stored_count += 1
            logger.info(
                f"Stored raw memory update {stored_count} in Qdrant",
                extra={
                    "message_id": message_uid,
                    "fact": (message_text if message_text else update_text)[:50],
                    "brief_fact": brief_fact[:50] if brief_fact else "",
                },
            )
    finally:
        await qdrant.close()
    
    logger.info(f"Stored {stored_count} raw memory updates total")
    return {"indexed_facts": indexed_facts}
