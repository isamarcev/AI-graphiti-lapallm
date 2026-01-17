"""
Node: Generate Solve Response
Extracts the final thought/decision from ReAct steps as the response.
"""

import logging
import re
from typing import Dict, Any, List
from agent.state import AgentState
from langsmith import traceable

logger = logging.getLogger(__name__)


def extract_references(text: str) -> List[str]:
    """Extract references from text matching [джерело: X] pattern."""
    # Match patterns like [джерело: 1], [джерело: abc123], etc.
    pattern = r'\[джерело:\s*([^\]]+)\]'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Deduplicate while preserving order
    seen = set()
    references = []
    for ref in matches:
        ref = ref.strip()
        if ref and ref not in seen:
            seen.add(ref)
            references.append(ref)
    
    return references


@traceable(name="generate_solve_response")
async def generate_solve_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Генерує відповідь для SOLVE режиму, використовуючи всі думки з ReAct.

    Args:
        state: Current agent state з react_steps

    Returns:
        State update with:
        - response: from ReAct agent
        - references: extracted from [джерело: X] patterns
        - reasoning: empty string
    """
    logger.info("=== Generate Solve Response Node ===")

    solve_response = state.get("solve_response", "")
    learn_response = state.get("learn_response", "")
    # Extract references from response
    references = extract_references(solve_response)

    response = solve_response + "\n\n" + learn_response
    logger.info(f"Extracted {len(references)} references: {references}")

    return {
        "response": response,
        "references": references,
        "reasoning": ""
    }