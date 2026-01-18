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
    indexed_facts = state.get("indexed_facts", [])
    conflicts = state.get("conflicts") or []
    
    response_parts = []
    
    # First part: list stored facts
    if indexed_facts:
        fact_list = ", ".join([f'"{item.get("brief_fact", "")}"' for item in indexed_facts if item.get("brief_fact")])
        if fact_list:
            response_parts.append(f"Зберіг наступні факти: {fact_list}.")
        else:
            response_parts.append("Інформацію не вдалося зберегти.")
    else:
        response_parts.append("Інформацію не вдалося зберегти.")
    
    # Second part: list conflicts if any
    if conflicts:
        conflict_list = ", ".join([f'"{fact}"' for _record_id, fact in conflicts])
        response_parts.append(
            f"Наступні факти суперечать вхідній інформації, тому позначені як недійсні: {conflict_list}."
        )
    
    # Concatenate response parts
    response = " ".join(response_parts)
    
    logger.info(
        "Generated learn response",
        extra={
            "indexed_facts_count": len(indexed_facts),
            "conflicts_count": len(conflicts),
            "response_length": len(response)
        }
    )
    
    return {"learn_response": response, "response": response}
