"""
Node for AI-powered context actualization and filtering.
Filters retrieved context items by relevance using Lapa LLM.
"""

import logging
from typing import Dict, Any, List
from langsmith import traceable

from agent.state import AgentState
from models.schemas import ContextActualizationResult, ContextRelevanceItem
from clients.llm_client import get_llm_client
from config.settings import settings

logger = logging.getLogger(__name__)


def _format_context_for_prompt(context_items: List[Dict[str, Any]]) -> str:
    """
    Format context items for the LLM prompt.

    Args:
        context_items: List of retrieved context dicts

    Returns:
        Formatted string with numbered context items
    """
    formatted = []
    for i, item in enumerate(context_items):
        content = item.get("content", "")
        score = item.get("score", 0.0)

        # Truncate long content to keep prompt manageable
        content_preview = content[:200] if len(content) > 200 else content

        formatted.append(f"[{i}] (score: {score:.2f}) {content_preview}")

    return "\n".join(formatted)


def _build_actualization_prompt(message_text: str, context_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build the Ukrainian prompt for context relevance assessment.

    Args:
        message_text: User query
        context_items: Retrieved context items

    Returns:
        List of message dicts for LLM
    """
    context_str = _format_context_for_prompt(context_items)

    system_prompt = """Ти експерт з оцінки релевантності контексту для точних відповідей на запити.

**ТВОЯ ЗАДАЧА:**
Оціни кожен елемент контексту відносно запиту користувача.

**КРИТИЧНІ ПРАВИЛА:**
1. Позначай як RELEVANT (is_relevant: true) ТІЛЬКИ якщо контекст БЕЗПОСЕРЕДНЬО допомагає відповісти на запит або виконати задачу
2. Нерелевантна інформація марнує токени та плутає агента - будь консервативним
3. Якщо сумніваєшся - краще позначити як нерелевантний
4. relevance_score має відображати впевненість: 1.0 = точно релевантний, 0.0 = точно нерелевантний

**ПРИКЛАДИ ОЦІНКИ:**

Запит: "Яка столиця України?"
[0] "Київ - столиця України"
→ is_relevant: true, relevance_score: 1.0, reason: "Прямо відповідає на запит про столицю"

[1] "Україна знаходиться в Європі"
→ is_relevant: false, relevance_score: 0.2, reason: "Не стосується питання про столицю"

Запит: "Що любить їсти Марія?"
[0] "Марія обожнює борщ"
→ is_relevant: true, relevance_score: 1.0, reason: "Прямо про улюблену їжу Марії"

[1] "Марія працює вчителькою"
→ is_relevant: false, relevance_score: 0.1, reason: "Професія не стосується їжі"

[2] "Борщ - традиційна українська страва"
→ is_relevant: false, relevance_score: 0.3, reason: "Загальна інформація про борщ, не про Марію"

Оціни кожен елемент незалежно та об'єктивно."""

    user_prompt = f"""**ЗАПИТ КОРИСТУВАЧА:**
{message_text}

**ДОСТУПНИЙ КОНТЕКСТ ДЛЯ ОЦІНКИ:**
{context_str}

Оціни релевантність кожного елемента контексту відносно запиту користувача."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


@traceable(name="actualize_context")
async def actualize_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Filter retrieved context using AI-based relevance assessment.

    Uses Lapa LLM with structured output to assess relevance of each context item
    relative to the user's query. Removes irrelevant items to improve token efficiency
    and reduce agent confusion.

    Args:
        state: AgentState with message_text and retrieved_context

    Returns:
        State update with filtered retrieved_context
    """
    logger.info("=== Actualize Context Node ===")

    # Extract data from state
    message_text = state.get("message_text", "")
    retrieved_context = state.get("retrieved_context", [])

    # Early exit: no context to filter
    if not retrieved_context:
        logger.info("No context to filter, passing through")
        return {"actualized_context": []}

    original_count = len(retrieved_context)
    logger.info(f"Filtering {original_count} context items for query: '{message_text[:100]}'")

    try:
        # Build prompt
        messages = _build_actualization_prompt(message_text, retrieved_context)

        # Call LLM with structured output
        llm_client = get_llm_client()
        result: ContextActualizationResult = await llm_client.generate_async(
            messages=messages,
            temperature=0.1,  # Low temperature for consistent filtering
            max_tokens=1000,
            response_format=ContextActualizationResult
        )

        logger.debug(f"LLM returned {result.total_relevant} relevant items out of {len(result.items)}")

        # Filter context based on LLM assessment
        filtered_context = []
        for item_assessment in result.items:
            idx = item_assessment.index

            # Validate index
            if idx < 0 or idx >= len(retrieved_context):
                logger.warning(f"Invalid index {idx} from LLM, skipping")
                continue

            # Keep if relevant and score above threshold
            if item_assessment.is_relevant and item_assessment.relevance_score >= settings.actualization_min_score:
                filtered_context.append(retrieved_context[idx])
                logger.debug(
                    f"✓ Kept item {idx} (score: {item_assessment.relevance_score:.2f}): {item_assessment.reason}"
                )
            else:
                logger.debug(
                    f"✗ Filtered out item {idx} (score: {item_assessment.relevance_score:.2f}): {item_assessment.reason}"
                )

        # Edge case: if all filtered out, keep top 1 by original Qdrant score
        if not filtered_context and original_count > 0:
            logger.warning(
                f"All {original_count} items filtered out, keeping top 1 by Qdrant score as fallback"
            )
            # Sort by score descending and take top 1
            sorted_context = sorted(retrieved_context, key=lambda x: x.get("score", 0.0), reverse=True)
            filtered_context = [sorted_context[0]]

        filtered_count = len(filtered_context)
        logger.info(f"Context filtering complete: {original_count} → {filtered_count} items")

        # Log token savings estimate
        items_removed = original_count - filtered_count
        if items_removed > 0:
            estimated_tokens_saved = items_removed * 150  # Rough estimate: 150 tokens per item
            logger.info(f"Estimated token savings: ~{estimated_tokens_saved} tokens")

        return {"actualized_context": filtered_context}

    except Exception as e:
        logger.error(f"Error during context actualization: {e}", exc_info=True)
        logger.warning("Falling back to unfiltered context due to error")

        # Graceful fallback: return original context
        return {"actualized_context": retrieved_context}
