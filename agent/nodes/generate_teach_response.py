"""
Node: Generate Teach Response
Генерує просту відповідь-підтвердження для TEACH режиму.
"""

import logging
from typing import Dict, Any
from agent.state import AgentState
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="generate_teach_response")
async def generate_teach_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Генерує відповідь для TEACH режиму.

    В TEACH режимі агент просто підтверджує що він зрозумів і зберіг знання.
    Не потрібен складний промпт - це просте підтвердження після store_knowledge.

    Args:
        state: Current agent state з message_text

    Returns:
        State update with:
        - response: підтвердження про збереження
        - references: [current message UID]
        - reasoning: None (не потрібен для teach)
    """
    logger.info("=== Generate Teach Response Node ===")

    message_uid = state["message_uid"]

    # Просте підтвердження що знання збережено
    response_text = (
        "Зрозуміло! Я зберіг цю інформацію у своїй базі знань. "
        "Тепер можу використати її для відповідей на запити."
    )

    logger.info(f"Generated teach confirmation for message {message_uid}")

    return {
        "response": response_text,
        "references": [message_uid],
        "reasoning": None
    }