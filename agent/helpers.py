"""
Helper functions для agent nodes.
Provides utility functions for ReAct reasoning, conflict detection, and message lookups.
"""

import logging
from typing import Optional, Tuple
from clients.llm_client import get_llm_client

logger = logging.getLogger(__name__)

def format_search_results(results: list) -> str:
    """
    Format Graphiti search results для observation в ReAct loop.
    
    Args:
        results: List of search results from Graphiti
    
    Returns:
        Formatted string for observation
    """
    if not results:
        return "Нічого не знайдено"
    
    formatted = []
    for i, result in enumerate(results, 1):
        # Handle different result formats
        content = result.get('content', '') or str(result)
        score = result.get('score', 0.0)
        
        # Truncate long content
        content_preview = content
        formatted.append(f"{i}. [score: {score:.2f}] {content_preview}")
    
    result_text = "\n".join(formatted)
    logger.debug(f"Formatted {len(results)} search results")
    return result_text
