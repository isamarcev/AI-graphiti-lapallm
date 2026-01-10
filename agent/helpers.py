"""
Helper functions для agent nodes.
Provides utility functions for ReAct reasoning, conflict detection, and message lookups.
"""

import re
import json
import logging
from typing import Optional, Tuple
from clients.llm_client import get_llm_client
from db.neo4j_helpers import get_message_store

logger = logging.getLogger(__name__)


async def get_message_uid_by_episode(episode_name: str) -> Optional[str]:
    """
    Wrapper для Neo4j lookup - знаходить message UID по episode name.
    
    Args:
        episode_name: Name of the Graphiti episode
    
    Returns:
        Message UID or None if not found
    """
    try:
        store = await get_message_store()
        return await store.get_message_uid_by_episode(episode_name)
    except Exception as e:
        logger.error(f"Error in get_message_uid_by_episode: {e}")
        return None


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


async def check_contradiction(
    text1: str,
    text2: str,
    threshold: float = 0.7
) -> Tuple[bool, float]:
    """
    Перевіряє чи два тексти суперечать один одному.
    
    Uses LLM для epistemic reasoning про конфлікт.
    Implements knowledge quality assessment - detecting contradictions
    with confidence scores.
    
    Args:
        text1: First text/fact
        text2: Second text/fact
        threshold: Minimum confidence to consider as conflict (default: 0.7)
    
    Returns:
        Tuple of (is_conflict: bool, confidence: float)
    
    Examples:
        "Київ - столиця України", "Харків - столиця України" -> (True, 0.95)
        "Олег любить каву", "Олег п'є чай" -> (False, 0.3)
    """
    llm = get_llm_client()
    
    prompt = f"""Проаналізуй чи ці два твердження суперечать одне одному.

Твердження 1: {text1}
Твердження 2: {text2}

Чи є протиріччя? Відповідь у форматі JSON:
{{
  "is_conflict": true або false,
  "confidence": число від 0.0 до 1.0,
  "explanation": "коротке пояснення"
}}

Правила:
- is_conflict = true тільки якщо твердження напряму суперечать
- confidence = твоя впевненість у висновку
- Якщо твердження просто про різні речі - це НЕ конфлікт"""
    
    try:
        response = await llm.generate_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        # Parse JSON response
        # Clean response if LLM added markdown
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        result = json.loads(clean_response)
        is_conflict = result.get("is_conflict", False)
        confidence = result.get("confidence", 0.5)
        explanation = result.get("explanation", "")
        
        logger.info(f"Contradiction check: {is_conflict} (conf: {confidence:.2f}) - {explanation}")
        
        # Apply threshold - не детектуємо low-confidence conflicts
        if confidence < threshold:
            logger.debug(f"Confidence {confidence} below threshold {threshold}, treating as no conflict")
            return False, confidence
        
        return is_conflict, confidence
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        logger.debug(f"Response was: {response}")
        # Fallback: conservative - не детектуємо конфлікт якщо parse failed
        return False, 0.0
    except Exception as e:
        logger.error(f"Error checking contradiction: {e}", exc_info=True)
        # Fallback: conservative - не детектуємо конфлікт якщо LLM failed
        return False, 0.0


def extract_used_sources(text: str) -> set:
    """
    Витягує номери використаних джерел з тексту відповіді.
    
    Шукає patterns типу [Джерело N] в тексті.
    
    Args:
        text: Response text containing source references
    
    Returns:
        Set of source indices used in the text
    
    Example:
        "Київ [Джерело 0] це столиця [Джерело 1]" -> {0, 1}
    """
    source_pattern = r'\[Джерело (\d+)\]'
    matches = re.findall(source_pattern, text)
    
    used_sources = set()
    for match in matches:
        try:
            idx = int(match)
            used_sources.add(idx)
        except ValueError:
            logger.warning(f"Invalid source index: {match}")
            continue
    
    logger.debug(f"Extracted {len(used_sources)} source references from text")
    return used_sources
