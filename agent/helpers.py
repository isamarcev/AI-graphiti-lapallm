"""
Helper functions для agent nodes.
Provides utility functions for ReAct reasoning, conflict detection, and message lookups.
"""

import re
import json
import logging
from typing import Optional, Tuple
from clients.llm_client import get_llm_client

logger = logging.getLogger(__name__)


def extract_search_query(thought: str) -> str:
    """
    Витягує search query з ReAct thought.
    
    Uses heuristics to extract the relevant search terms from the agent's
    reasoning step.
    
    Args:
        thought: ReAct thought/reasoning text
    
    Returns:
        Extracted search query string
    
    Examples:
        "Потрібно знайти інформацію про Київ" -> "Київ"
        "Шукати 'столиця України'" -> "столиця України"
    """
    # Simple heuristic: extract quoted text first
    quoted = re.findall(r'["\'](.+?)["\']', thought)
    if quoted:
        query = quoted[0]
        logger.debug(f"Extracted query from quotes: {query}")
        return query
    
    # Fallback: keywords після "про", "шукати", "знайти"
    keywords = ["про", "шукати", "знайти", "пошук", "інформацію про", "information about"]
    for keyword in keywords:
        if keyword in thought.lower():
            parts = thought.lower().split(keyword)
            if len(parts) > 1:
                # Take first 3 words after keyword
                words = parts[1].strip().split()[:3]
                query = " ".join(words)
                logger.debug(f"Extracted query after '{keyword}': {query}")
                return query
    
    # Last resort: return whole thought (truncated)
    query = thought[:100]
    logger.debug(f"Using full thought as query (truncated): {query}")
    return query


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
        content_preview = content[:200] + "..." if len(content) > 200 else content
        formatted.append(f"{i}. [score: {score:.2f}] {content_preview}")
    
    result_text = "\n".join(formatted)
    logger.debug(f"Formatted {len(results)} search results")
    return result_text



def extract_used_sources(text: str) -> set:
    """
    DEPRECATED: Use extract_message_uids_from_text instead.
    
    Old function that extracted numeric indices.
    Kept for backward compatibility.
    """
    logger.warning("extract_used_sources is deprecated, use extract_message_uids_from_text")
    return set()


def extract_message_uids_from_text(text: str) -> set:
    """
    Витягує message UIDs використаних джерел з тексту відповіді.
    
    Шукає patterns типу [msg-XXX] або [test-msg-XXX] в тексті.
    
    Args:
        text: Response text containing source references
    
    Returns:
        Set of message UIDs used in the text
    
    Example:
        "Київ [msg-001] це столиця [msg-002]" -> {"msg-001", "msg-002"}
        "Харків [test-msg-005]" -> {"test-msg-005"}
    """
    # Pattern для message UIDs: [msg-XXX] або [test-msg-XXX] або [UUID]
    uid_pattern = r'\[((?:test-)?msg-\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]'
    matches = re.findall(uid_pattern, text, re.IGNORECASE)
    
    used_uids = set(matches)
    logger.debug(f"Extracted {len(used_uids)} message UID references from text: {used_uids}")
    return used_uids
