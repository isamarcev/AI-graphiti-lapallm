"""
Node 4: Conflict Resolution
Asks user to resolve detected conflicts.
"""

import logging
from typing import Dict, Any

from agent.state import AgentState
from db.neo4j_helpers import get_message_store
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="resolve_conflict")
async def resolve_conflict_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Form conflict resolution question.

    This node:
    1. Takes the first detected conflict
    2. Saves it to DB (conflicts + pending_resolutions)
    3. Forms a question for the user
    4. Returns response and STOPS (waits for user resolution)

    Args:
        state: Current agent state

    Returns:
        State update with response (question) and pending_resolution_id
    """
    logger.info("=== Resolve Conflict Node ===")

    conflicts = state.get("conflicts", [])
    if not conflicts:
        logger.warning("No conflicts to resolve")
        return {
            "response": "Помилка: конфлікт не знайдено",
            "conflict_resolved": True
        }

    # Take first conflict
    conflict = conflicts[0]
    logger.info(f"Resolving conflict: {conflict['description']}")

    # Form question for user
    question = (
        f"⚠️ Виявлено протиріччя:\n\n"
        f"**Стара інформація** (повідомлення {conflict['old_msg_uid']}):\n"
        f"{conflict['old_content']}\n\n"
        f"**Нова інформація** (повідомлення {conflict['new_msg_uid']}):\n"
        f"{conflict['new_content']}\n\n"
        f"Як вирішити цей конфлікт?\n"
        f"1. Використати нове значення\n"
        f"2. Залишити старе\n"
        f"3. Обидва правильні (різні контексти)\n\n"
        f"Будь ласка, уточніть яка інформація правильна."
    )

    return {
        "response": question,
        "references": [conflict["old_msg_uid"], conflict["new_msg_uid"]],
        "conflict_resolved": False
    }
