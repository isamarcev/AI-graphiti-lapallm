import logging
from typing import Dict, Any
from agent.state import AgentState
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="generate_learn_response")
async def generate_learn_response_node(state: AgentState) -> Dict[str, Any]:
    message_text = state.get("message_text", "")
    conflicts = state.get("conflicts") or []

    if not conflicts:
        response = f'Факт "{message_text}" збережено.'
    else:
        conflict_facts = ", ".join([fact for _msg_id, fact in conflicts])
        response = (
            f'Факт "{message_text}" збережено. '
            f'Факти "{conflict_facts}" помічені як недійсні.'
        )

    logger.info("Generated learn response", extra={"has_conflicts": bool(conflicts)})
    return {"response": response}