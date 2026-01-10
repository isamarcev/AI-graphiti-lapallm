"""
Node 6: Context Retrieval
Знаходить relevant context для SOLVE задач через Graphiti hybrid search.
"""

import logging
from typing import Dict, Any
from agent.state import AgentState
from agent.helpers import get_message_uid_by_episode
from clients.graphiti_client import get_graphiti_client
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="retrieve_context")
async def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Hybrid search в Graphiti для SOLVE path.
    
    Implements knowledge retrieval with quality metrics:
    - Semantic + BM25 hybrid search
    - Relevance scores для кожного result
    - Source message UIDs для references
    
    Args:
        state: Current agent state with message_text (query)
    
    Returns:
        State update with retrieved_context containing:
        - content: retrieved knowledge
        - source_msg_uid: reference to source message
        - timestamp: when knowledge was stored
        - score: relevance score
    """
    logger.info("=== Retrieve Context Node ===")
    logger.info(f"Query: {state['message_text'][:100]}...")
    
    graphiti = await get_graphiti_client()
    
    try:
        # Search knowledge graph with hybrid search
        search_results = await graphiti.search(
            query=state["message_text"],
            limit=settings.graphiti_search_limit
        )
        
        logger.info(f"Graphiti returned {len(search_results)} results")
        
        # Format з source message UIDs
        retrieved_context = []
        
        for result in search_results:
            # Extract content from result
            content = result.get('content', '') or str(result)
            score = result.get('score', 1.0)
            timestamp = result.get('timestamp') or result.get('created_at')
            
            # Skip empty results
            if not content or len(content.strip()) < 5:
                logger.debug("Skipping empty result")
                continue
            
            # Get source message UID
            # Try different fields where episode name might be stored
            episode_name = (
                result.get('episode_name') or 
                result.get('source') or
                result.get('name')
            )
            source_msg_uid = None
            
            if episode_name:
                source_msg_uid = await get_message_uid_by_episode(episode_name)
                if not source_msg_uid:
                    logger.debug(f"Could not find message UID for episode: {episode_name}")
            
            retrieved_context.append({
                "content": content,
                "source_msg_uid": source_msg_uid or "unknown",
                "timestamp": timestamp,
                "score": score
            })
            
            logger.debug(
                f"Retrieved: {content[:100]}... "
                f"(score: {score:.2f}, source: {source_msg_uid or 'unknown'})"
            )
        
        logger.info(f"Retrieved {len(retrieved_context)} context items with valid content")
        
        if not retrieved_context:
            logger.warning("No relevant context found in knowledge base")
        
        return {"retrieved_context": retrieved_context}
        
    except Exception as e:
        logger.error(f"Error retrieving context: {e}", exc_info=True)
        return {"retrieved_context": []}
