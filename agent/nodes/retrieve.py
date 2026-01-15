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
from models.schemas import RetrievedContext
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
        
        # Convert to RetrievedContext format
        retrieved_context: List[RetrievedContext] = []
        
        for hit in search_results:
            payload = hit.get("payload", {})
            score = hit.get("score", 0.0)
            
            # Extract fact and source information
            fact = payload.get("fact", "")
            message_id = payload.get("messageid") or payload.get("message_id") or "unknown"
            timestamp = payload.get("timestamp")
            
            if fact:
                context_item = RetrievedContext(
                    content=fact,
                    source_msg_uid=message_id,
                    timestamp=timestamp,
                    score=score
                )
                retrieved_context.append(context_item)
                logger.debug(f"Retrieved: {fact[:50]}... (score: {score:.3f})")
        
        logger.info(f"Retrieved {len(retrieved_context)} context items")
        
        # Convert to dict format for state storage
        context_dicts = [
            {
                "content": ctx.content,
                "source_msg_uid": ctx.source_msg_uid,
                "timestamp": ctx.timestamp,
                "score": ctx.score
            }
            for ctx in retrieved_context
        ]
        
        return {
            "retrieved_context": context_dicts
        }
        
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}", exc_info=True)
        return {"retrieved_context": []}
        
    finally:
        await qdrant.close()
