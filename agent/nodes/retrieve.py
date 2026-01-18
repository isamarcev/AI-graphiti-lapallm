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
    Retrieve initial context from Qdrant using optimized search queries.

    Uses search_queries from query_analysis if available, otherwise falls back
    to message_text. Performs multi-query retrieval for better coverage.

    Args:
        state: Current agent state with message_text and optional query_analysis

    Returns:
        State update with retrieved_context list
    """
    logger.info("=== Context Retriever Node ===")

    message_text = state.get("message_text", "")
    query_analysis = state.get("query_analysis")

    if not message_text:
        logger.warning("No message_text to retrieve context for")
        return {"retrieved_context": []}

    # Визначаємо пошукові запити
    search_queries = state.get("search_queries", [])
    if not search_queries:
        logger.info("No search queries available, using original message_text")
        search_queries = [message_text]

    logger.info(f"Search queries: {search_queries}")

    # Initialize clients
    embedder = HostedQwenEmbedder()
    qdrant = QdrantClient()
    await qdrant.initialize()

    try:
        all_results = []
        seen_message_ids = set()  # Deduplicate by source message ID

        # Виконуємо пошук для кожного запиту
        for idx, query in enumerate(search_queries):
            logger.info(f"Query {idx + 1}/{len(search_queries)}: '{query}'")

            # Generate embedding for this query
            query_vector = await embedder.create(query)

            # Search for relevant context
            # Берем по 3 результати для кожного запиту замість 5 для одного
            # Щоб загальна кількість була ~ 5-9 після дедуплікації
            top_k_per_query = 3 if len(search_queries) > 1 else 5

            search_results = await qdrant.search_similar(
                query_vector=query_vector,
                top_k=top_k_per_query,
                only_relevant=True,
            )

            logger.info(f"  Found {len(search_results)} results for this query")
            all_results.extend(search_results)

        logger.info(f"Total raw results: {len(all_results)}")
        
        # Build context dicts with deduplication
        context_dicts = []

        for hit in all_results:
            payload = hit.get("payload", {})
            score = hit.get("score", 0.0)

            # Extract all fields from payload
            fact = payload.get("fact", "")
            brief_fact = payload.get("brief_fact", "")
            message_id = payload.get("message_id") or "unknown"
            timestamp = payload.get("timestamp")

            # Deduplicate: skip if we already have context from this message
            if message_id in seen_message_ids:
                logger.debug(f"Skipping duplicate from message {message_id}")
                continue

            if fact:
                seen_message_ids.add(message_id)
                context_dicts.append({
                    "content": fact,
                    "brief_fact": brief_fact,
                    "message_id": message_id,
                    "timestamp": timestamp,
                    "score": score
                })
                logger.debug(f"Retrieved: {fact[:50]}... (score: {score:.3f})")

        # Sort by score descending and limit to top 10
        context_dicts.sort(key=lambda x: x["score"], reverse=True)
        context_dicts = context_dicts[:10]

        logger.info(f"Retrieved {len(context_dicts)} unique context items (after deduplication)")
        
        return {
            "retrieved_context": context_dicts
        }
        
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}", exc_info=True)
        return {"retrieved_context": []}
        
    finally:
        await qdrant.close()
