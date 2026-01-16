import logging
import asyncio
from typing import Any, Dict, List, Optional

from agent.state import AgentState
from clients.hosted_embedder import HostedQwenEmbedder
from clients.qdrant_client import QdrantClient
from langsmith import traceable

logger = logging.getLogger(__name__)


async def store_single_fact(
        indexed: Dict[str, Any],
        message_uid: str,
        embedder: HostedQwenEmbedder,
        qdrant: QdrantClient,
        semaphore: asyncio.Semaphore,
        idx: int
) -> Optional[List[float]]:
    """
    Обрабатывает один факт: создает вектор и сохраняет в Qdrant.
    Возвращает вектор в случае успеха или None в случае ошибки.
    """
    fact_text = indexed.get("fact") or ""
    description = indexed.get("description")
    examples = indexed.get("examples")

    if not fact_text:
        logger.warning(f"Skipping indexed fact {idx} with empty fact field")
        return None

    # Если описания нет, используем сам факт для эмбеддинга, чтобы избежать ошибок
    text_to_embed = description if description else fact_text

    async with semaphore:
        try:
            # 1. Параллельно создаем вектор
            vector = await embedder.create(text_to_embed)

            # 2. Параллельно пишем в базу
            await qdrant.insert_record(
                vector=vector,
                fact=fact_text,
                message_id=message_uid,
                is_relevant=True,
                payload={"description": description, "examples": examples},
            )

            logger.info(
                f"Stored indexed fact {idx} in Qdrant (Async)",
                extra={
                    "message_id": message_uid,
                    "fact": fact_text[:50],
                    "has_description": bool(description),
                }
            )
            return vector

        except Exception as e:
            logger.error(f"Failed to store fact {idx}: {e}")
            return None


@traceable(name="store_indexed_facts")
async def store_indexed_facts_node(state: AgentState) -> Dict[str, Any]:
    """
    Persist indexed facts to Qdrant using parallel execution.
    """
    indexed_facts = state.get("indexed_facts") or []

    if not indexed_facts:
        logger.warning("store_indexed_fact_node: no indexed_facts to store")
        return {}

    logger.info(f"Starting parallel storage of {len(indexed_facts)} facts")

    embedder = HostedQwenEmbedder()
    qdrant = await QdrantClient().initialize()

    # Ограничиваем количество одновременных записей (10 - безопасное число)
    semaphore = asyncio.Semaphore(10)

    tasks = []

    try:
        # Формируем задачи
        for i, indexed in enumerate(indexed_facts, 1):
            task = store_single_fact(
                indexed=indexed,
                message_uid=state.get("message_uid", ""),
                embedder=embedder,
                qdrant=qdrant,
                semaphore=semaphore,
                idx=i
            )
            tasks.append(task)

        # Запускаем всё параллельно
        # results будет списком векторов (или None) в том же порядке, что и indexed_facts
        results = await asyncio.gather(*tasks)

        # Подсчитываем успешные сохранения
        successful_vectors = [v for v in results if v is not None]
        stored_count = len(successful_vectors)

        # Получаем последний успешный вектор для возврата (как в оригинальной логике)
        last_vector = successful_vectors[-1] if successful_vectors else []

    finally:
        await qdrant.close()

    logger.info(f"Stored {stored_count} indexed facts total (Parallel)")

    return {"message_embedding": last_vector}