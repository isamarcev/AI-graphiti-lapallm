"""
Enhanced Context Retriever Node з Multi-Source Queries та RRF Fusion

Покращення:
1. Використання key_entities, information_needs разом з search_queries
2. Reciprocal Rank Fusion для комбінування results
3. Краща дедуплікація та ранжування
4. Опціональний cross-encoder reranking (якщо ENABLE_RERANKER=true)
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
) -> List[tuple[str, float]]:
    """
    Reciprocal Rank Fusion algorithm для комбінування результатів з різних queries.

    Formula: score(d) = Σ 1 / (k + rank_i(d))

    Переваги RRF:
    - Не потрібна нормалізація similarity scores
    - Robust до outliers
    - State-of-the-art в RAG системах

    Args:
        results_per_query: Список результатів для кожного запиту
                          Кожен результат має {"id": ..., "score": ..., "payload": ...}
        k: Константа для згладжування (зазвичай 60)

    Returns:
        Список кортежів (doc_id, rrf_score) відсортований за спаданням score

    Example:
        Query 1: [doc_A (rank 1), doc_B (rank 2)]
        Query 2: [doc_B (rank 1), doc_A (rank 3)]

        RRF scores:
        - doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        - doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

        Result: [(doc_B, 0.0325), (doc_A, 0.0323)]
    """
    scores = defaultdict(float)

    for query_idx, results in enumerate(results_per_query):
        logger.debug(f"Processing query {query_idx + 1}/{len(results_per_query)}: {len(results)} results")

        for rank, result in enumerate(results, start=1):
            doc_id = str(result["id"])  # Ensure string ID

            # RRF formula
            rrf_contribution = 1.0 / (k + rank)
            scores[doc_id] += rrf_contribution

            logger.debug(
                f"  Doc {doc_id[:8]}... rank={rank} → RRF contribution={rrf_contribution:.4f}"
            )

    # Sort by RRF score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"RRF fusion: {len(sorted_results)} unique documents")
    if sorted_results:
        logger.info(f"Top RRF score: {sorted_results[0][1]:.4f}")

    return sorted_results


def find_document_by_id(
    results_per_query: List[List[Dict[str, Any]]],
    doc_id: str
) -> Dict[str, Any] | None:
    """
    Знайти оригінальний документ по ID серед усіх результатів.

    Args:
        results_per_query: Всі результати пошуків
        doc_id: ID документа для пошуку

    Returns:
        Документ з найвищим original score або None
    """
    candidates = []

    for results in results_per_query:
        for doc in results:
            if str(doc["id"]) == doc_id:
                candidates.append(doc)

    if not candidates:
        return None

    # Повертаємо документ з найвищим similarity score
    # (може бути кілька копій з різних queries)
    return max(candidates, key=lambda x: x["score"])


@traceable(name="retrieve_context_enhanced")
async def retrieve_context_enhanced_node(state: AgentState) -> Dict[str, Any]:
    """
    Enhanced context retrieval з multi-source queries та RRF fusion.

    Покращення:
    1. Використовує search_queries + key_entities + information_needs
    2. RRF для кращого комбінування results
    3. Збільшений top_k з кращою дедуплікацією

    Args:
        state: AgentState з message_text та query_analysis

    Returns:
        State update з retrieved_context list
    """
    logger.info("=== Enhanced Context Retriever Node ===")

    message_text = state.get("message_text", "")
    query_analysis = state.get("query_analysis")

    if not message_text:
        logger.warning("No message_text to retrieve context for")
        return {"retrieved_context": []}

    # ========================================
    # 1. ФОРМУВАННЯ QUERIES З РІЗНИХ ДЖЕРЕЛ
    # ========================================

    queries = []

    # A. Search queries (основні, найважливіші)
    if query_analysis and query_analysis.get("search_queries"):
        search_queries = query_analysis["search_queries"]
        queries.extend(search_queries)
        logger.info(f"✓ Added {len(search_queries)} search_queries")

    # B. Required tools/methods (інструменти/методи згадані в запиті)
    if query_analysis and query_analysis.get("required_tools_or_methods"):
        tools = query_analysis["required_tools_or_methods"]
        queries.extend(tools)
        logger.info(f"✓ Added {len(tools)} required_tools_or_methods")

    # C. Key entities (топ-3 для точних match)
    if query_analysis and query_analysis.get("key_entities"):
        entities = query_analysis["key_entities"][:3]  # обмежуємо 3
        queries.extend(entities)
        logger.info(f"✓ Added {len(entities)} key_entities")

    # D. Information needs (топ-3 для контексту)
    if query_analysis and query_analysis.get("information_needs"):
        needs = query_analysis["information_needs"][:3]  # збільшили з 2 до 3
        queries.extend(needs)
        logger.info(f"✓ Added {len(needs)} information_needs")

    # D. Original message_text (як fallback для пропущених контекстів)
    # Додаємо завжди, щоб не пропустити важливу інформацію (напр. "Мавка")
    if query_analysis:
        queries.append(message_text)

    # Fallback: якщо нічого немає, використовуємо оригінальний текст
    if not queries:
        queries = [message_text]
        logger.warning("No query_analysis available, using original message_text")

    # Дедуплікація queries (зберігаючи порядок)
    queries_before_dedup = len(queries)
    queries = list(dict.fromkeys(queries))  # Remove duplicates while preserving order

    if queries_before_dedup > len(queries):
        logger.info(
            f"Removed {queries_before_dedup - len(queries)} duplicate queries. "
            f"Total unique queries: {len(queries)}"
        )
    else:
        logger.info(f"Total queries to search: {len(queries)}")

    logger.info(f"Unique queries: {queries}")

    # ========================================
    # 2. ВИКОНАННЯ ПОШУКУ ДЛЯ КОЖНОГО ЗАПИТУ
    # ========================================

    embedder = HostedQwenEmbedder()
    qdrant = QdrantClient()
    await qdrant.initialize()

    try:
        results_per_query = []

        for idx, query in enumerate(queries):
            logger.info(f"Query {idx + 1}/{len(queries)}: '{query}'")

            # Generate embedding
            query_vector = await embedder.create(query)

            # Search (збільшили top_k з 3 до 5)
            search_results = await qdrant.search_similar(
                query_vector=query_vector,
                top_k=5,  # було 3, стало 5
                only_relevant=True,
            )

            logger.info(f"  → Found {len(search_results)} results")
            results_per_query.append(search_results)

        total_raw_results = sum(len(r) for r in results_per_query)
        logger.info(f"Total raw results: {total_raw_results}")

        # ========================================
        # 3. RRF FUSION
        # ========================================

        # Використовуємо RRF замість простого extend + sort
        fused_results = reciprocal_rank_fusion(results_per_query, k=60)

        logger.info(f"After RRF fusion: {len(fused_results)} unique documents")

        # ========================================
        # 4. ДЕДУПЛІКАЦІЯ ТА ФОРМАТУВАННЯ
        # ========================================

        context_dicts = []
        seen_message_ids = set()

        for doc_id, rrf_score in fused_results[:15]:  # беремо топ-15 для дедуплікації
            # Знайти оригінальний документ
            doc = find_document_by_id(results_per_query, doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found in original results")
                continue

            payload = doc.get("payload", {})
            message_id = payload.get("message_id", "unknown")

            # Дедуплікація по message_id
            if message_id in seen_message_ids:
                logger.debug(f"Skipping duplicate message_id: {message_id}")
                continue

            fact = payload.get("fact", "")
            if not fact:
                logger.debug(f"Skipping document with empty fact")
                continue

            seen_message_ids.add(message_id)

            context_dicts.append({
                "content": fact,
                "source_msg_uid": message_id,
                "timestamp": payload.get("timestamp"),
                "score": rrf_score,  # RRF score замість similarity
                "original_similarity": doc.get("score", 0.0),  # зберігаємо оригінальний
                "description": payload.get("description", ""),
                "examples": payload.get("examples", ""),
            })

            logger.debug(
                f"Retrieved: {fact[:50]}... "
                f"(RRF={rrf_score:.4f}, sim={doc.get('score', 0.0):.3f})"
            )

            # Limit to top 10 after deduplication
            if len(context_dicts) >= 10:
                break

        logger.info(f"Final retrieved contexts: {len(context_dicts)}")

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

        return {"retrieved_context": context_dicts}

    except Exception as e:
        logger.error(f"Error during enhanced retrieval: {e}", exc_info=True)
        return {"retrieved_context": []}

    finally:
        await qdrant.close()


# ========================================
# UTILITY FUNCTIONS
# ========================================

def log_retrieval_stats(results_per_query: List[List[Dict]], fused_results: List[tuple]):
    """
    Логування статистики пошуку для аналізу.
    """
    logger.info("=== Retrieval Statistics ===")

    # Per-query stats
    for idx, results in enumerate(results_per_query):
        if results:
            avg_score = sum(r["score"] for r in results) / len(results)
            logger.info(f"Query {idx + 1}: {len(results)} results, avg_score={avg_score:.3f}")

    # Fusion stats
    if fused_results:
        logger.info(f"After fusion: {len(fused_results)} unique docs")
        logger.info(f"Top-3 RRF scores: {[f'{s:.4f}' for _, s in fused_results[:3]]}")


def compare_retrieval_methods(
    simple_results: List[Dict],
    enhanced_results: List[Dict]
) -> Dict[str, Any]:
    """
    Порівняння простого та enhanced retrieval.

    Для A/B тестування та аналізу ефективності.
    """
    simple_ids = {r["source_msg_uid"] for r in simple_results}
    enhanced_ids = {r["source_msg_uid"] for r in enhanced_results}

    overlap = simple_ids & enhanced_ids
    only_simple = simple_ids - enhanced_ids
    only_enhanced = enhanced_ids - simple_ids

    return {
        "overlap_count": len(overlap),
        "overlap_percentage": len(overlap) / max(len(simple_ids), 1) * 100,
        "only_in_simple": len(only_simple),
        "only_in_enhanced": len(only_enhanced),
        "simple_total": len(simple_ids),
        "enhanced_total": len(enhanced_ids),
    }