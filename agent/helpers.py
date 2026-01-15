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


async def check_contradiction(
    text1: str,
    text2: str,
    threshold: float = 0.7
) -> Tuple[bool, float, str]:
    """
    Перевіряє 5 типів конфліктів між фактами.

    Uses LLM для epistemic reasoning про конфлікт.
    Implements knowledge quality assessment - detecting contradictions
    with confidence scores and conflict type classification.

    Args:
        text1: First text/fact
        text2: Second text/fact
        threshold: Minimum confidence to consider as conflict (default: 0.7)

    Returns:
        Tuple of (is_conflict: bool, confidence: float, conflict_type: str)

    Conflict Types:
        1. direct - прямі протиріччя (Київ vs Харків)
        2. temporal - зміна в часі (раніше любив, зараз ні)
        3. contextual - залежить від контексту (сам vs з друзями)
        4. degree - різниця в ступені (любить vs обожнює)
        5. partial - часткове протиріччя (любить каву vs не любить напої)

    Examples:
        "Київ - столиця України", "Харків - столиця України" -> (True, 0.95, "direct")
        "Раніше любив каву", "Зараз не люблю каву" -> (False, 1.0, "temporal")
        "Олег любить каву", "Олег п'є чай" -> (False, 0.3, "none")
    """
    llm = get_llm_client()

    prompt = f"""Проаналізуй ці два твердження на ВСІХ рівнях конфліктів.

**Твердження 1:** {text1}
**Твердження 2:** {text2}

**ТИПИ КОНФЛІКТІВ:**

1. **DIRECT** - прямі взаємовиключні факти
   Приклад: "Київ столиця України" vs "Харків столиця України"

2. **TEMPORAL** - зміна в часі (це OK, НЕ завжди конфлікт!)
   "Раніше любив каву" vs "Зараз не люблю каву" = НЕ конфлікт (природня зміна)
   "Любив каву" vs "Ніколи не любив каву" = КОНФЛІКТ (суперечить минулому)

3. **CONTEXTUAL** - залежить від обставин (обидва можуть бути правдою)
   "Люблю плавати сам" vs "Люблю плавати з друзями" = НЕ конфлікт

4. **DEGREE** - різниця в інтенсивності (уточнення, не конфлікт)
   "Люблю каву" vs "Обожнюю каву" = НЕ конфлікт (уточнення)

5. **PARTIAL** - одне включає інше
   "Люблю каву" vs "Не люблю напої" = partial conflict

**FEW-SHOT EXAMPLES:**

Приклад 1:
Твердження 1: "Київ столиця України"
Твердження 2: "Харків столиця України"
→ {{"is_conflict": true, "conflict_type": "direct", "confidence": 1.0, "explanation": "Країна не може мати дві столиці"}}

Приклад 2:
Твердження 1: "Олег любить плавати сам"
Твердження 2: "Олег любить плавати з друзями"
→ {{"is_conflict": false, "conflict_type": "contextual", "confidence": 0.9, "explanation": "Обидва можуть бути правдою в різних контекстах"}}

Приклад 3:
Твердження 1: "Я веган"
Твердження 2: "Сьогодні я їв стейк"
→ {{"is_conflict": true, "conflict_type": "direct", "confidence": 0.95, "explanation": "Веган не їсть м'ясо за визначенням"}}

Приклад 4:
Твердження 1: "Раніше я працював в Google"
Твердження 2: "Зараз я не працюю в Google"
→ {{"is_conflict": false, "conflict_type": "temporal", "confidence": 1.0, "explanation": "Природня зміна в часі, не конфлікт"}}

Приклад 5:
Твердження 1: "Я завжди працював ТІЛЬКИ в Google"
Твердження 2: "Раніше я працював в Microsoft"
→ {{"is_conflict": true, "conflict_type": "direct", "confidence": 0.9, "explanation": "Якщо ЗАВЖДИ тільки Google, то не міг працювати в Microsoft"}}

**ПРАВИЛА:**
- TEMPORAL changes з явним часом = НЕ конфлікт
- CONTEXTUAL differences = НЕ конфлікт якщо можуть співіснувати
- DEGREE differences = НЕ конфлікт якщо уточнення
- Confidence < 0.7 = не детектуємо як конфлікт

Відповідай JSON:
{{"is_conflict": bool, "conflict_type": "direct"|"temporal"|"contextual"|"degree"|"partial"|"none", "confidence": 0.0-1.0, "explanation": "чому є/немає конфлікту"}}"""
    
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
        conflict_type = result.get("conflict_type", "unknown")
        explanation = result.get("explanation", "")

        logger.info(
            f"Conflict check: {is_conflict} (type: {conflict_type}, conf: {confidence:.2f}) - {explanation}"
        )

        # Apply threshold - не детектуємо low-confidence conflicts
        if confidence < threshold:
            logger.debug(f"Confidence {confidence} below threshold {threshold}, treating as no conflict")
            return False, confidence, conflict_type

        return is_conflict, confidence, conflict_type
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        logger.debug(f"Response was: {response}")
        # Fallback: conservative - не детектуємо конфлікт якщо parse failed
        return False, 0.0, "error"
    except Exception as e:
        logger.error(f"Error checking contradiction: {e}", exc_info=True)
        # Fallback: conservative - не детектуємо конфлікт якщо LLM failed
        return False, 0.0, "error"


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
