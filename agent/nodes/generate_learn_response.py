import logging
from typing import Dict, Any
from agent.state import AgentState
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="generate_learn_response")
async def generate_learn_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate response for pure LEARN scenarios.
    
    IMPORTANT: This node is called ONLY when there are no solve tasks.
    For mixed messages (learn + solve), this node is SKIPPED to save tokens,
    and only the solve response is generated.
    
    Args:
        state: Current agent state
        
    Returns:
        State update with response text
    """
    logger.info("=== Generate Learn Response Node ===")
    
    # Get the data from state
    memory_updates = state.get("memory_updates", [])
    conflicts = state.get("conflicts") or []
    stored_count = state.get("stored_memories_count", len(memory_updates))
    
    # Generate response based on what was learned
    if stored_count == 0:
        response = "Інформацію не вдалося зберегти."
        logger.warning("No memories stored")
    elif stored_count == 1:
        # Single fact learned
        if not conflicts:
            response = "Інформацію збережено."
        else:
            conflict_facts = ", ".join([f'"{fact}"' for _msg_id, fact in conflicts])
            response = (
                f"Інформацію збережено. "
                f"Попередні факти {conflict_facts} помічені як недійсні."
            )
    else:
        # Multiple facts learned
        if not conflicts:
            response = f"Збережено {stored_count} фактів."
        else:
            conflict_count = len(conflicts)
            response = (
                f"Збережено {stored_count} фактів. "
                f"{conflict_count} попередніх фактів помічено як недійсні."
            )
    
    logger.info(
        "Generated learn response",
        extra={
            "stored_count": stored_count,
            "has_conflicts": bool(conflicts),
            "response_length": len(response)
        }
    )
    
    return {"response": response}