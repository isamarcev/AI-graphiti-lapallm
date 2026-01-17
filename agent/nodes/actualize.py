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

    system_prompt = """Ти експерт з фільтрації контексту для AI-агента, який відповідає на запити користувачів.

**МЕТА ФІЛЬТРАЦІЇ:**
Агент отримає ЛИШЕ релевантний контекст для генерації відповіді. Нерелевантна інформація марнує токени та може спантеличити агента.

**КОЛИ КОНТЕКСТ РЕЛЕВАНТНИЙ:**
✓ Контекст містить ПРЯМУ відповідь на запит
✓ Контекст містить КЛЮЧОВУ інформацію, необхідну для відповіді
✓ Контекст стосується КОНКРЕТНОЇ сутності/персони з запиту

**КОЛИ КОНТЕКСТ НЕРЕЛЕВАНТНИЙ:**
✗ Загальна інформація, яка не стосується конкретного запиту
✗ Інформація про ІНШИХ людей/сутності/задачі (не тих, про кого питають)
✗ Інформація про ІНШІ аспекти (питають про їжу, а контекст про роботу)
✗ Побічний контекст (може бути цікавим, але не відповідає на запит)

**ВАЖЛИВО:**
- У разі сумнівів - позначай як НЕРЕЛЕВАНТНИЙ (is_relevant: false)
- Будь консервативним: краще менше контексту, але точного
- Оцінюй кожен елемент НЕЗАЛЕЖНО від інших

**ПРИКЛАДИ:**

Запит: "Яка столиця України?"
[0] "Київ - столиця України" → is_relevant: true (пряма відповідь)
[1] "Україна знаходиться в Європі" → is_relevant: false (загальна інформація)
[2] "Київ - найбільше місто України" → is_relevant: false (не про столицю)

Запит: "Що любить їсти Марія?"
[0] "Марія обожнює борщ і вареники" → is_relevant: true (пряма відповідь про їжу Марії)
[1] "Марія працює вчителькою" → is_relevant: false (про роботу, не про їжу)
[2] "Борщ - традиційна українська страва" → is_relevant: false (загальна інформація про борщ)
[3] "Сестра Марії теж любить борщ" → is_relevant: false (про іншу людину)

Запит: "Де працює Олег?"
[0] "Олег працює в IT-компанії" → is_relevant: true (пряма відповідь)
[1] "Олег живе в Києві" → is_relevant: false (про місце проживання, не роботу)
[2] "Друг Олега працює в банку" → is_relevant: false (про друга, не про Олега)

Запит: "Розкажи про хобі Ірини"
[0] "Ірина захоплюється фотографією" → is_relevant: true (пряма інформація про хобі)
[1] "Ірина часто їздить на природу" → is_relevant: false (може бути пов'язано, але не конкретно про хобі)
[2] "Фотографія - популярне хобі" → is_relevant: false (загальна інформація)"""

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
    return {"actualized_context": retrieved_context}

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

            # Keep if relevant
            if item_assessment.is_relevant:
                filtered_context.append(retrieved_context[idx])
                logger.debug(f"✓ Kept item {idx} as relevant")
            else:
                logger.debug(f"✗ Filtered out item {idx} as not relevant")

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

        return {"actualized_context": filtered_context}

    except Exception as e:
        logger.error(f"Error during context actualization: {e}", exc_info=True)
        logger.warning("Falling back to unfiltered context due to error")

        # Graceful fallback: return original context
        return {"actualized_context": retrieved_context}
