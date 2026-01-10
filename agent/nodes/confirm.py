"""
Node 6: Generate Confirmation
Generates a short confirmation message after extracting facts from a teaching message.
"""

import logging
from typing import Dict, Any
from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="generate_confirmation")
async def generate_confirmation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 6: Generate confirmation message for teaching.

    Creates a short, natural confirmation in Ukrainian that acknowledges
    the learned facts. This confirmation will be combined with the user's
    message to form a proper conversation for Graphiti.

    Args:
        state: Current agent state with extracted_facts

    Returns:
        State update with confirmation_text
    """
    logger.info("=== Generate Confirmation Node ===")

    extracted_facts = state.get("extracted_facts", [])
    fact_count = len(extracted_facts)

    logger.info(f"Generating confirmation for {fact_count} extracted facts")

    # If no facts were extracted, use a neutral confirmation
    if fact_count == 0:
        confirmation = "Дякую за інформацію. Я її зберіг."
        logger.info("No facts extracted, using neutral confirmation")
        return {
            "confirmation_text": confirmation
        }

    llm = get_llm_client()

    # Format facts for the prompt
    facts_list = "\n".join([
        f"- {fact['subject']} {fact['relation']} {fact['object']}"
        for fact in extracted_facts
    ])

    prompt = f"""Ти — асистент в TABULA RASA режимі (починаєш з нульовими знаннями).
Щойно вивчив наступні факти:

{facts_list}

Згенеруй КОРОТКЕ (1-2 речення) підтвердження українською, що демонструє РОЗУМІННЯ структури/процесу/концепту.

**ПРАВИЛА:**
- Покажи що ЗРОЗУМІВ структуру, не просто "запам'ятав"
- Якщо це процедура/алгоритм - перекажи своїми словами логіку
- Якщо це структура/синтаксис - покажи розуміння компонентів
- Якщо це факт - підтверди що записав
- Будь природним і коротким (1-2 речення)

**ПРИКЛАДИ:**

Особиста інформація:
❌ "Дякую, запам'ятав"
✅ "Дякую! Тепер я знаю, що тебе звати Олексій і ти любиш каву"

Процедура/алгоритм:
❌ "Записав про процес"
✅ "Зрозумів! Спочатку порівнюємо елементи, потім міняємо місцями. Це ітеративний процес"

Структура/синтаксис:
❌ "Дякую за інформацію"
✅ "Зрозумів! Структура складається з трьох частин: ключове слово, потім ім'я, потім значення"

Визначення/константа:
❌ "Запам'ятав"
✅ "Добре, тепер я знаю що pi = 3.14. Це числова константа"

Твоє підтвердження:"""

    try:
        response = await llm.generate_async(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )

        confirmation = response.strip()
        logger.info(f"Generated confirmation: {confirmation}")

        return {
            "confirmation_text": confirmation
        }

    except Exception as e:
        logger.error(f"Error generating confirmation: {e}", exc_info=True)
        # Fallback to simple confirmation
        fallback = f"Дякую! Я вивчив {fact_count} {'факт' if fact_count == 1 else 'факти' if fact_count < 5 else 'фактів'}."
        logger.info(f"Using fallback confirmation: {fallback}")
        return {
            "confirmation_text": fallback
        }