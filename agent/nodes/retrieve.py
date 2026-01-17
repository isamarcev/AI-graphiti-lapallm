"""
Node: Context Retriever
Performs initial context retrieval BEFORE ReAct loop.
This gives the agent starting context to reason with.
"""

import logging
from typing import Dict, Any, List

from agent.state import AgentState
from clients.qdrant_client import QdrantClient
from clients.hosted_embedder import HostedQwenEmbedder
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="retrieve_context")
async def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve initial context from Qdrant based on the task message.
    
    This happens BEFORE the ReAct loop, giving the agent starting context.
    The ReAct loop can still perform additional searches if needed.
    
    Args:
        state: Current agent state with message_text (the task)
        
    Returns:
        State update with retrieved_context list
    """
    logger.info("=== Context Retriever Node ===")
    
    message_text = state.get("message_text", "")
    
    if not message_text:
        logger.warning("No message_text to retrieve context for")
        return {"retrieved_context": []}
    
    logger.info(f"Retrieving context for: {message_text[:100]}...")
    
    # Initialize clients
    embedder = HostedQwenEmbedder()
    qdrant = QdrantClient()
    await qdrant.initialize()
    
    try:
        # Generate embedding for the task
        query_vector = await embedder.create(message_text)
        logger.info("Generated query embedding")
        
        # Search for relevant context
        search_results = await qdrant.search_similar(
            query_vector=query_vector,
            top_k=5,  # Get top 5 most relevant facts
            only_relevant=True,  # Only return facts marked as relevant
        )
        
        logger.info(f"Found {len(search_results)} search results")
        
        # Build context dicts with all payload fields
        context_dicts = []
        
        for hit in search_results:
            payload = hit.get("payload", {})
            score = hit.get("score", 0.0)
            
            # Extract all fields from payload
            fact = payload.get("fact", "")
            message_id = payload.get("message_id") or "unknown"
            timestamp = payload.get("timestamp")
            description = payload.get("description", "")
            examples = payload.get("examples", "")
            
            if fact:
                context_dicts.append({
                    "content": fact,
                    "source_msg_uid": message_id,
                    "timestamp": timestamp,
                    "score": score,
                    "description": description,
                    "examples": examples
                })
                logger.debug(f"Retrieved: {fact[:50]}... (score: {score:.3f})")
        
        logger.info(f"Retrieved {len(context_dicts)} context items")
        
        return {
            "retrieved_context": context_dicts
        }
        
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}", exc_info=True)
        return {"retrieved_context": []}
        
    finally:
        await qdrant.close()
