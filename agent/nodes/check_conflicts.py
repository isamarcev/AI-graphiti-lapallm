

import logging
import json
from typing import Any, Dict, List, Tuple

from langsmith import traceable

from agent.state import AgentState
from clients.qdrant_client import QdrantClient
from clients.hosted_embedder import HostedQwenEmbedder
from clients.llm_client import get_llm_client
from config.settings import settings

logger = logging.getLogger(__name__)


def _get_embedder():
    return HostedQwenEmbedder()


def _extract_similar_facts(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    similar_facts = []
    for item in results:
        payload = item.get("payload") or {}
        fact = payload.get("fact")
        message_id = payload.get("message_id")
        # Use record_id from payload if available, otherwise use the Qdrant point ID
        record_id = payload.get("record_id") or str(item.get("id"))
        if fact and record_id:
            similar_facts.append({"fact": fact, "message_id": message_id, "record_id": record_id})
    return similar_facts


@traceable(name="check_conflicts")
async def check_conflicts_node(state: AgentState) -> Dict[str, Any]:
    """
    Check conflicts for each memory_update individually.
    For each update:
      1. Vectorize it
      2. Find 3 nearest neighbors in Qdrant
      3. Use LLM to detect conflicts
      4. Mark conflicting neighbors as irrelevant immediately
    """

    logger.info("=== Check Conflicts Node (LEARN) ===")
    
    memory_updates = state.get("memory_updates", [])
    if not memory_updates:
        logger.info("No memory_updates to check for conflicts")
        return {"conflicts": []}

    logger.info(f"Processing {len(memory_updates)} memory update(s) for conflict check")

    embedder = _get_embedder()
    llm = get_llm_client()
    qdrant = await QdrantClient().initialize()

    total_conflicts_resolved = 0
    all_conflicts: List[Tuple[str, str]] = []  # (record_id, fact) for reporting

    try:
        for idx, update_text in enumerate(memory_updates, 1):
            logger.info(f"Checking conflicts for memory_update {idx}/{len(memory_updates)}")

            # 1. Vectorize the update
            vector = await embedder.create(update_text)

            # 2. Find 3 nearest neighbors
            results = await qdrant.search_similar(
                query_vector=vector,
                top_k=3,
                only_relevant=True,
            )

            similar_facts = _extract_similar_facts(results)
            logger.info("*****"*10)

            logger.info(results)
            logger.info("*****"*10)
            logger.info(f"Found {len(similar_facts)} similar facts for update {idx}")

            if not similar_facts:
                continue

            message_fact_map: Dict[str, str] = {
                item["record_id"]: item["fact"] for item in similar_facts
            }

            # 3. Build prompt for conflict checking
            facts_lines = "\n".join(
                [
                    f"- record_id={item['record_id']}: {item['fact']}"
                    for item in similar_facts
                ]
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
            

            try:
                llm_result = await llm.generate_async(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=settings.temperature,
                )
                
                content = llm_result if isinstance(llm_result, str) else str(llm_result)
                try:
                    parsed = json.loads(content)
                    conflicts = parsed.get("conflicts", [])
                    if not isinstance(conflicts, list):
                        conflicts = []
                except Exception:
                    conflicts = []
            except Exception as e:
                logger.error(f"Conflict LLM check failed for update {idx}: {e}")
                conflicts = []

            # 4. Mark conflicting records as irrelevant immediately
            for cid in conflicts:
                if not isinstance(cid, str):
                    continue
                cid_clean = cid.strip()
                if not cid_clean:
                    continue
                
                if cid_clean in message_fact_map:
                    try:
                        await qdrant.set_relevance_by_record_id(record_id=cid_clean, is_relevant=False)
                        logger.info(f"Marked record_id={cid_clean} as irrelevant (conflict with update {idx})")
                        total_conflicts_resolved += 1
                        # Track for reporting
                        all_conflicts.append((cid_clean, message_fact_map[cid_clean]))
                    except Exception as e:
                        logger.error(f"Failed to mark record_id={cid_clean} as irrelevant: {e}")
                else:
                    logger.warning(
                        f"Conflict id from LLM not found in similar facts; skipping",
                        extra={"record_id": cid_clean},
                    )

    finally:
        await qdrant.close()

    logger.info(f"Resolved {total_conflicts_resolved} conflicts total")

    return {"conflicts": all_conflicts}
