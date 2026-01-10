"""
Node 4a: Auto-Resolve Conflicts
Automatically accepts new information, replacing old conflicting data.
For Tabula Rasa agent - new information takes precedence.
"""

import logging
from typing import Dict, Any

from agent.state import AgentState
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="auto_resolve_conflict")
async def auto_resolve_conflict_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4a: Automatically resolve conflicts by accepting new information.
    
    Implements Tabula Rasa principle: new information automatically replaces old.
    Example: "pi = 4" automatically replaces "pi = 3.14"
    
    This is the PRIMARY conflict resolution path for direct conflicts.
    Only escalate to manual resolution (resolve_conflict_node) for critical uncertainties.
    
    Args:
        state: Current agent state with conflicts
    
    Returns:
        State update with:
        - response: notification about update
        - auto_resolved: True (marks as auto-resolved)
        - resolution: "accept_new" (indicates resolution strategy)
        - references: source UIDs for transparency
    """
    logger.info("=== Auto-Resolve Conflict Node ===")
    
    conflicts = state.get("conflicts", [])
    if not conflicts:
        logger.warning("No conflicts to auto-resolve")
        return {
            "response": "Інформацію збережено",
            "auto_resolved": True,
            "resolution": "no_conflict"
        }
    
    # Take first conflict
    conflict = conflicts[0]
    conflict_type = conflict.get("conflict_type", "direct")
    logger.info(f"Auto-resolving {conflict_type} conflict")
    
    # Auto-accept new information
    old_uid = conflict.get("old_msg_uid", "unknown")
    new_uid = conflict.get("new_msg_uid", "unknown")
    old_content = conflict.get("old_content", "")
    new_content = conflict.get("new_content", "")
    
    # Generate transparent notification
    response = f"""✓ **Інформацію оновлено**

**Було** (повідомлення `{old_uid}`):
> {old_content}

**Тепер** (повідомлення `{new_uid}`):
> {new_content}

Я оновив свої знання відповідно до нової інформації."""
    
    # Generate confirmation_text для збереження в Graphiti
    # Format similar to generate_confirmation but shows update
    confirmation_text = f"Зрозумів! Оновив інформацію: тепер {new_content}."
    
    logger.info(f"Auto-resolved: old={old_uid}, new={new_uid}")
    
    return {
        "response": response,
        "confirmation_text": confirmation_text,  # For Graphiti episode
        "auto_resolved": True,
        "resolution": "accept_new",
        "references": [old_uid, new_uid],
        "conflict_resolved": True
    }
