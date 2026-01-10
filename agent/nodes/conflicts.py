"""
Node 3: Conflict Detection
Детектує conflicts між новими facts і існуючими knowledge.
Implements knowledge quality assessment (paper).
"""

import logging
from typing import Dict, Any
from agent.state import AgentState
from agent.helpers import check_contradiction, get_message_uid_by_episode
from clients.graphiti_client import get_graphiti_client
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="check_conflicts")
async def check_conflicts_node(state: AgentState) -> Dict[str, Any]:
    """
    Для кожного extracted fact:
    1. Hybrid search в Graphiti для схожих facts
    2. LLM-based contradiction detection
    3. Збирання conflicts з epistemic metrics (confidence, similarity)
    
    Knowledge quality focus: detecting contradictions and inconsistencies.
    
    Implements bidirectional knowledge flow:
    - Detects when new knowledge conflicts with existing
    - Enables agent to ask user for clarification
    
    Args:
        state: Current agent state with extracted_facts
    
    Returns:
        State update with conflicts list
    """
    logger.info("=== Check Conflicts Node ===")
    
    extracted_facts = state.get("extracted_facts", [])
    if not extracted_facts:
        logger.info("No facts to check")
        return {"conflicts": []}
    
    graphiti = await get_graphiti_client()
    conflicts = []
    
    for fact in extracted_facts:
        # Format fact як текст для search
        fact_text = f"{fact['subject']} {fact['relation']} {fact['object']}"
        logger.info(f"Checking fact: {fact_text}")
        
        try:
            # Search existing knowledge
            search_results = await graphiti.search(
                query=fact_text,
                limit=3
            )
            
            if not search_results:
                logger.debug(f"No existing knowledge found for: {fact_text}")
                continue
            
            # Check each result for contradiction
            for result in search_results:
                old_content = result.get('content', '') or str(result)
                
                # Skip if too short
                if len(old_content.strip()) < 5:
                    continue
                
                # Epistemic reasoning про conflict (5 типів)
                is_conflict, confidence, conflict_type = await check_contradiction(
                    fact_text, old_content
                )

                if is_conflict:
                    # Extract source message UID
                    # Try different fields where episode name might be stored
                    episode_name = (
                        result.get('episode_name') or 
                        result.get('source') or
                        result.get('name')
                    )
                    old_msg_uid = None
                    
                    if episode_name:
                        old_msg_uid = await get_message_uid_by_episode(episode_name)
                    
                    conflict = {
                        "old_msg_uid": old_msg_uid or "unknown",
                        "new_msg_uid": state["message_uid"],
                        "old_content": old_content,
                        "new_content": fact_text,
                        "conflict_type": conflict_type,  # NEW: direct/temporal/contextual/degree/partial
                        "description": f"Конфлікт ({conflict_type}): '{old_content}' vs '{fact_text}'",
                        "confidence": confidence  # Epistemic metric
                    }
                    
                    conflicts.append(conflict)
                    logger.warning(f"Conflict detected: {conflict['description']}")
                else:
                    logger.debug(f"No conflict between '{fact_text}' and '{old_content}'")
        
        except Exception as e:
            logger.error(f"Error checking fact '{fact_text}': {e}", exc_info=True)
            continue
    
    logger.info(f"Total conflicts found: {len(conflicts)}")
    return {"conflicts": conflicts}
