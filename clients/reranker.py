"""
Reranker Client - CPU-based cross-encoder for context reranking.

Використовує sentence-transformers cross-encoder для швидкого reranking
без витрат на LLM токени.

Моделі (від швидкої до точної):
- cross-encoder/ms-marco-MiniLM-L-6-v2  (~22M params, ~50ms/batch on CPU)
- cross-encoder/ms-marco-MiniLM-L-12-v2 (~33M params, ~100ms/batch on CPU)
- cross-encoder/ms-marco-TinyBERT-L-2-v2 (~4M params, ~20ms/batch on CPU)

Використання:
    from clients.reranker import get_reranker, rerank_contexts

    # Опціонально (автоматично викликається при першому rerank)
    reranker = get_reranker()

    # Основний метод
    reranked = await rerank_contexts(
        query="Як працює Мавка?",
        contexts=[{"content": "...", "source_msg_uid": "123"}, ...],
        top_k=5
    )

ВАЖЛИВО: Цей модуль опціональний і контролюється через ENABLE_RERANKER=true
"""

import logging
from typing import Dict, Any, List, Optional
from functools import lru_cache

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Глобальний стан для lazy loading
_reranker_model = None
_reranker_available = None


def _check_reranker_available() -> bool:
    """Перевіряє чи доступний sentence-transformers."""
    global _reranker_available

    if _reranker_available is not None:
        return _reranker_available

    try:
        from sentence_transformers import CrossEncoder
        _reranker_available = True
        logger.info("sentence-transformers available for reranking")
    except ImportError:
        _reranker_available = False
        logger.warning(
            "sentence-transformers not installed. Reranker disabled. "
            "Install with: pip install sentence-transformers"
        )

    return _reranker_available


def get_reranker():
    """
    Lazy-load CrossEncoder model.

    Returns:
        CrossEncoder instance або None якщо недоступний/вимкнений
    """
    global _reranker_model

    settings = get_settings()

    # Перевіряємо чи увімкнено
    if not settings.enable_reranker:
        logger.debug("Reranker disabled via settings")
        return None

    # Перевіряємо чи доступний
    if not _check_reranker_available():
        return None

    # Lazy load моделі
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        model_name = settings.reranker_model
        logger.info(f"Loading reranker model: {model_name}")

        try:
            _reranker_model = CrossEncoder(
                model_name,
                max_length=512,  # Обмежуємо для швидкості
                device="cpu"     # Явно CPU
            )
            logger.info(f"Reranker model loaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            return None

    return _reranker_model


async def rerank_contexts(
    query: str,
    contexts: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    min_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Rerank contexts using cross-encoder model.

    Args:
        query: User query
        contexts: List of context dicts with 'content' field
        top_k: Number of top results to return (default from settings)
        min_score: Minimum score threshold (default from settings)

    Returns:
        Reranked list of contexts with added 'rerank_score' field

    Note:
        Якщо reranker недоступний/вимкнений, повертає оригінальний список.
    """
    settings = get_settings()

    # Defaults from settings
    if top_k is None:
        top_k = settings.reranker_top_k
    if min_score is None:
        min_score = settings.reranker_min_score

    # Skip якщо немає контексту
    if not contexts:
        return contexts

    # Get reranker (може бути None якщо вимкнено/недоступно)
    reranker = get_reranker()

    if reranker is None:
        logger.debug("Reranker not available, returning original contexts")
        return contexts

    logger.info(f"Reranking {len(contexts)} contexts for query: '{query[:50]}...'")

    try:
        # Підготовка пар (query, document) для cross-encoder
        pairs = []
        for ctx in contexts:
            content = ctx.get("content") or ctx.get("fact") or ""
            pairs.append((query, content))

        # Cross-encoder scoring (синхронний, але швидкий на CPU)
        # Використовуємо convert_to_numpy для швидкості
        scores = reranker.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Нормалізація scores до [0, 1] за допомогою sigmoid
        # (cross-encoder scores можуть бути будь-якими числами)
        import numpy as np
        normalized_scores = 1 / (1 + np.exp(-scores))

        # Додаємо scores до контекстів
        scored_contexts = []
        for ctx, score in zip(contexts, normalized_scores):
            ctx_copy = ctx.copy()
            ctx_copy["rerank_score"] = float(score)
            scored_contexts.append(ctx_copy)

        # Сортуємо за rerank_score (descending)
        scored_contexts.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Фільтруємо за min_score та top_k
        filtered = [
            ctx for ctx in scored_contexts
            if ctx["rerank_score"] >= min_score
        ][:top_k]

        # Логування результатів
        if filtered:
            logger.info(
                f"Reranking complete: {len(contexts)} -> {len(filtered)} contexts. "
                f"Score range: {filtered[-1]['rerank_score']:.3f} - {filtered[0]['rerank_score']:.3f}"
            )
        else:
            logger.warning(
                f"All contexts filtered out by reranker (min_score={min_score}). "
                f"Falling back to top-{min(2, len(scored_contexts))} by score."
            )
            # Safety fallback: повертаємо хоча б 2 найкращих
            filtered = scored_contexts[:min(2, len(scored_contexts))]

        return filtered

    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        # Graceful fallback: повертаємо оригінальний список
        return contexts


def is_reranker_enabled() -> bool:
    """Check if reranker is enabled and available."""
    settings = get_settings()
    return settings.enable_reranker and _check_reranker_available()
