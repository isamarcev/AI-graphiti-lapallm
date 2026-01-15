

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


@traceable(name="check_conflicts")
async def check_conflicts_node(state: AgentState) -> Dict[str, Any]:

    logger.info("=== Check Conflicts Node (LEARN) ===")
    message_text = state["message_text"]

    # Embed message
    embedder = _get_embedder()
    vector = await embedder.create(message_text)

    # Qdrant search (only is_relevant=True)
    qdrant = await QdrantClient().initialize()
    results = await qdrant.search_similar(
        query_vector=vector,
        top_k=3,
        only_relevant=True,
    )

    await qdrant.close()

    similar_facts = _extract_similar_facts(results)
    logger.info(f"Found {len(similar_facts)} similar facts")

    if not similar_facts:
        return {"conflicts": []}

    message_fact_map: Dict[str, str] = {
        item["record_id"]: item["fact"] for item in similar_facts
    }

    # Build prompt for conflict checking
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
        f"Новий факт: \"{message_text}\"\n\n"
        f"Збережені факти:\n{facts_lines}\n\n"
        "Відповідай JSON: {\"conflicts\": [\"record_id1\", \"record_id2\", ...]}"
    )
    llm = get_llm_client()

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    logger.info(f"Conflict LLM prompt: {message}")
    logger.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    try:
        llm_result = await llm.generate_async(
            messages=message,
            temperature=0.0,
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
        logger.error(f"Conflict LLM check failed: {e}")
        conflicts = []

    # Normalize conflicts to list of (record_id, fact) tuples for AgentState
    normalized_conflicts: List[Tuple[str, str]] = []
    for cid in conflicts:
        if not isinstance(cid, str):
            continue
        cid_clean = cid.strip()
        if not cid_clean:
            continue
        fact_text = message_fact_map.get(cid_clean)
        if fact_text:
            normalized_conflicts.append((cid_clean, fact_text))
        else:
            logger.warning(
                "Conflict id from LLM not found in similar facts payload; skipping",
                extra={"record_id": cid_clean},
            )

    return {
        "conflicts": normalized_conflicts,
        "message_embedding": vector,
    }
