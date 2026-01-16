import asyncio
import logging
import json
from typing import Any, Dict, List, Tuple

from langsmith import traceable

from agent.state import AgentState
from clients.qdrant_client import QdrantClient
from clients.hosted_embedder import HostedQwenEmbedder
from clients.llm_client import get_llm_client

logger = logging.getLogger(__name__)


def _get_embedder():
    return HostedQwenEmbedder()


def _extract_similar_facts(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    similar_facts = []
    for item in results:
        payload = item.get("payload") or {}
        fact = payload.get("fact")
        message_id = payload.get("messageid") or payload.get("message_id")
        record_id = payload.get("record_id")
        if fact and record_id:
            similar_facts.append({"fact": fact, "message_id": message_id, "record_id": record_id})
    return similar_facts


async def process_single_update(
        update_text: str,
        idx: int,
        embedder: Any,
        llm: Any,
        qdrant: Any,
        semaphore: asyncio.Semaphore
) -> List[Tuple[str, str]]:
    """
    Обробляє один update_text. Повертає список знайдених конфліктів (record_id, fact).
    """
    async with semaphore:  # Обмежуємо кількість одночасних запитів
        try:
            logger.info(f"Start checking update {idx}")

            # 1. Vectorize
            vector = await embedder.create(update_text)

            # 2. Find neighbors
            results = await qdrant.search_similar(
                query_vector=vector,
                top_k=3,
                only_relevant=True,
            )

            similar_facts = _extract_similar_facts(results)
            if not similar_facts:
                return []

            message_fact_map = {item["record_id"]: item["fact"] for item in similar_facts}

            # 3. Build Prompt
            facts_lines = "\n".join(
                [f"- record_id={item['record_id']}: {item['fact']}" for item in similar_facts]
            )

            system_prompt = (
                "Ти інструмент для виявлення суперечностей.\n"
                "Отримуєш новий факт від користувача та список раніше збережених фактів.\n"
                "Поверни JSON зі списком record_id тих фактів, які суперечать новому факту.\n"
                "Якщо суперечностей немає — поверни порожній список."
            )
            user_prompt = (
                f"Новий факт: \"{update_text}\"\n\n"
                f"Збережені факти:\n{facts_lines}\n\n"
                "Відповідай JSON: {\"conflicts\": [\"record_id1\", \"record_id2\", ...]}"
            )

            # LLM Call
            try:
                llm_result = await llm.generate_async(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                content = str(llm_result)
                parsed = json.loads(content)
                conflicts_ids = parsed.get("conflicts", [])
                if not isinstance(conflicts_ids, list):
                    conflicts_ids = []
            except Exception as e:
                logger.error(f"Conflict LLM check failed for update {idx}: {e}")
                return []

            # 4. Resolve conflicts
            resolved_conflicts = []
            for cid in conflicts_ids:
                if not isinstance(cid, str): continue
                cid_clean = cid.strip()
                if not cid_clean: continue

                if cid_clean in message_fact_map:
                    try:
                        # Тут важливо: Qdrant клієнт має бути thread-safe або підтримувати async
                        await qdrant.set_relevance_by_record_id(record_id=cid_clean,
                                                                is_relevant=False)
                        logger.info(
                            f"Marked record_id={cid_clean} as irrelevant (conflict with update {idx})")
                        resolved_conflicts.append((cid_clean, message_fact_map[cid_clean]))
                    except Exception as e:
                        logger.error(f"Failed to mark record_id={cid_clean} as irrelevant: {e}")

            return resolved_conflicts

        except Exception as e:
            logger.error(f"Critical error processing update {idx}: {e}")
            return []


@traceable(name="check_conflicts")
async def check_conflicts_node(state: AgentState) -> Dict[str, Any]:
    logger.info("=== Check Conflicts Node (LEARN) ===")

    memory_updates = state.get("memory_updates", [])
    if not memory_updates:
        return {"conflicts": []}

    embedder = _get_embedder()
    llm = get_llm_client()
    qdrant = await QdrantClient().initialize()

    # Створюємо семафор, наприклад, на 5 одночасних запитів
    semaphore = asyncio.Semaphore(5)

    try:
        tasks = []
        for idx, update_text in enumerate(memory_updates, 1):
            task = process_single_update(
                update_text=update_text,
                idx=idx,
                embedder=embedder,
                llm=llm,
                qdrant=qdrant,
                semaphore=semaphore
            )
            tasks.append(task)

        # Запускаємо всі задачі паралельно і чекаємо завершення
        results_list = await asyncio.gather(*tasks)

        # Об'єднуємо результати (flatten list)
        all_conflicts = [item for sublist in results_list for item in sublist]

    finally:
        await qdrant.close()

    logger.info(f"Resolved {len(all_conflicts)} conflicts total")
    return {"conflicts": all_conflicts}
