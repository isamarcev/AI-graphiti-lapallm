"""
Node: Context Retriever
Performs initial context retrieval BEFORE ReAct loop.
This gives the agent starting context to reason with.

Опціонально використовує reranker для фільтрації (якщо ENABLE_RERANKER=true).
"""

import logging
from typing import Dict, Any, List
from collections import defaultdict

from agent.state import AgentState
from clients.qdrant_client import QdrantClient
from clients.hosted_embedder import HostedQwenEmbedder
from clients.reranker import rerank_contexts, is_reranker_enabled
from langsmith import traceable

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_per_query: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[tuple]:
    """
    Reciprocal Rank Fusion algorithm for combining multiple ranked lists.

    RRF formula: score(d) = Σ 1 / (k + rank_i(d))

    Args:
        results_per_query: List of result lists, one per query
        k: Constant for RRF formula (default: 60)

    Returns:
        List of (doc_id, rrf_score) tuples sorted by score descending
    """
    scores = defaultdict(float)

    for results in results_per_query:
        for rank, result in enumerate(results, start=1):
            doc_id = str(result["id"])
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def find_document_by_id(
    results_per_query: List[List[Dict[str, Any]]],
    doc_id: str
) -> Dict[str, Any] | None:
    """
    Find the original document by its ID across all query results.

    Args:
        results_per_query: List of result lists from all queries
        doc_id: Document ID to find

    Returns:
        Original document dict or None if not found
    """
    for results in results_per_query:
        for doc in results:
            if str(doc["id"]) == doc_id:
                return doc
    return None


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

    # Формуємо queries з різних джерел для кращого coverage
    queries = []
    num_search_queries = 0
    num_entities = 0
    num_needs = 0

    if query_analysis:
        # A. Search queries (основні оптимізовані запити)
        if query_analysis.get("search_queries"):
            search_queries = query_analysis["search_queries"]
            queries.extend(search_queries)
            num_search_queries = len(search_queries)

        # B. Key entities (топ-3 для пошуку по сутностях)
        if query_analysis.get("key_entities"):
            entities = query_analysis["key_entities"][:3]
            queries.extend(entities)
            num_entities = len(entities)

        # C. Information needs (топ-3 для пошуку по потребах)
        if query_analysis.get("information_needs"):
            needs = query_analysis["information_needs"][:3]
            queries.extend(needs)
            num_needs = len(needs)

        # D. Original message_text (як fallback для захоплення пропущених контекстів)
        # Додаємо в кінець, щоб не домінувати над аналізованими queries
        queries.append(message_text)

    # Fallback: якщо немає жодних queries
    if not queries:
        queries = [message_text]
        logger.info("No query analysis available, using original message_text")
    else:
        # Дедуплікація queries (зберігаючи порядок)
        queries_before_dedup = len(queries)
        queries = list(dict.fromkeys(queries))  # Remove duplicates while preserving order

        if queries_before_dedup > len(queries):
            logger.info(
                f"Removed {queries_before_dedup - len(queries)} duplicate queries. "
                f"Total unique queries: {len(queries)} "
                f"(search_queries={num_search_queries}, "
                f"entities={num_entities}, "
                f"needs={num_needs})"
            )
        else:
            logger.info(
                f"Total queries: {len(queries)} "
                f"(search_queries={num_search_queries}, "
                f"entities={num_entities}, "
                f"needs={num_needs})"
            )

    logger.info(f"Unique queries: {queries}")

    # Initialize clients
    embedder = HostedQwenEmbedder()
    qdrant = QdrantClient()
    await qdrant.initialize()

    try:
        results_per_query = []

        # Виконуємо пошук для кожного запиту і зберігаємо результати окремо
        for idx, query in enumerate(queries):
            logger.info(f"Query {idx + 1}/{len(queries)}: '{query}'")

            # Generate embedding for this query
            query_vector = await embedder.create(query)

            # Search for relevant context
            # Збільшуємо top_k з 3 до 5 для кращого coverage
            search_results = await qdrant.search_similar(
                query_vector=query_vector,
                top_k=5,
                only_relevant=True,
            )

            logger.info(f"  Found {len(search_results)} results for this query")
            results_per_query.append(search_results)

        logger.info(f"Total raw results: {sum(len(r) for r in results_per_query)}")

        # Застосовуємо RRF fusion для комбінування результатів
        fused_results = reciprocal_rank_fusion(results_per_query, k=60)
        logger.info(f"After RRF fusion: {len(fused_results)} unique documents")
        
        # Build context dicts з дедуплікацією по message_id
        context_dicts = []
        seen_message_ids = set()

        # Проходимо по топ-15 RRF результатів для дедуплікації
        for doc_id, rrf_score in fused_results[:15]:
            # Знаходимо оригінальний документ
            doc = find_document_by_id(results_per_query, doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found in results")
                continue

            payload = doc.get("payload", {})

            # Extract all fields from payload
            fact = payload.get("fact", "")
            message_id = payload.get("message_id") or "unknown"
            timestamp = payload.get("timestamp")
            description = payload.get("description", "")
            examples = payload.get("examples", "")

            # Дедуплікація: пропускаємо якщо вже є контекст з цього message
            if message_id in seen_message_ids:
                logger.debug(f"Skipping duplicate from message {message_id}")
                continue

            if fact:
                seen_message_ids.add(message_id)
                context_dicts.append({
                    "content": fact,
                    "source_msg_uid": message_id,
                    "timestamp": timestamp,
                    "score": rrf_score,  # RRF score замість similarity score
                    "description": description,
                    "examples": examples
                })
                logger.debug(f"Retrieved: {fact[:50]}... (RRF score: {rrf_score:.4f})")

            # Зупиняємося після 10 унікальних документів
            if len(context_dicts) >= 10:
                break

        logger.info(f"Final results after deduplication: {len(context_dicts)} unique context items")

        # === OPTIONAL RERANKING ===
        # Якщо увімкнено reranker, фільтруємо контекст за допомогою cross-encoder
        # Це замінює LLM-based actualize_context без витрат токенів
        if is_reranker_enabled() and context_dicts:
            logger.info("Applying cross-encoder reranking...")
            context_dicts = await rerank_contexts(
                query=message_text,
                contexts=context_dicts
            )
            logger.info(f"After reranking: {len(context_dicts)} contexts")

        return {
            "retrieved_context": context_dicts
        }

    except Exception as e:
        logger.error(f"Error during context retrieval: {e}", exc_info=True)
        return {"retrieved_context": []}

    finally:
        await qdrant.close()
