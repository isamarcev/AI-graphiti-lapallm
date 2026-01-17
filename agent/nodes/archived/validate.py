"""
Node: Response Validator
Mock validator for solve responses.
Future: Can add quality checks and retry logic.
"""

import logging
from typing import Dict, Any

from agent.state import AgentState
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="validate_response")
async def validate_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Validate the generated solve response.
    
    Current implementation: Mock validator that always passes.
    Future enhancements:
    - Check if response uses only information from context (no hallucinations)
    - Verify referenced_sources are valid
    - Check has_sufficient_info flag
    - Implement retry logic if validation fails (up to max_retries)
    
    Args:
        state: Current agent state with response
        
    Returns:
        State update with validation_status
    """
    logger.info("=== Response Validator Node (Mock) ===")
    
    response = state.get("response", "")
    references = state.get("references", [])
    
    # Basic checks
    if not response:
        logger.warning("Empty response detected")
        return {"validation_status": "invalid", "validation_error": "Empty response"}
    
    if not references:
        logger.warning("No references in response")
        # Note: This might be valid for "no context" scenarios
    
    logger.info(f"Response length: {len(response)} chars")
    logger.info(f"References count: {len(references)}")
    
    # For now: always pass validation
    # Future: Add actual validation logic here
    logger.info("Validation passed (mock)")
    
    return {
        "validation_status": "valid"
    }
