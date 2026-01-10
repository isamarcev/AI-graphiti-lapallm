"""
Node 1: Intent Classification
Determines if message is TEACH (learning) or SOLVE (task).
"""

import logging
from typing import Dict, Any
from pydantic import BaseModel

from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    intent: str  # "teach" or "solve"
    confidence: float
    reasoning: str


@traceable(name="classify_intent")
async def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Classify intent of user message.

    Determines whether the user is:
    - TEACH: Providing information/facts to learn
    - SOLVE: Asking to solve a task/question

    Args:
        state: Current agent state

    Returns:
        State update with intent
    """
    logger.info("=== Classify Intent Node ===")
    logger.info(f"Message: {state['message_text'][:100]}...")

    llm = get_llm_client()

    system_prompt = """🚫 **TABULA RASA РЕЖИМ**:
Ти починаєш з НУЛЬОВИМИ знаннями про предметну область.
Класифікуй повідомлення БЕЗ припущень про домен (це може бути ЩО ЗАВГОДНО).

Визнач чи користувач:

1. **TEACH (навчання)** - дає інформацію, факти, описує, пояснює, навчає
   Приклади (універсальні домени):
   - Особиста інформація: "Моє ім'я Олег", "У мене алергія на арахіс"
   - Географія/історія: "Столиця України - Київ", "Війна закінчилася в 1945"
   - Процедури: "Щоб спекти хліб, спочатку замісити тісто"
   - Структурована інформація: "У цій системі команда має формат: дія об'єкт параметри"
   - Визначення: "Значення pi дорівнює 3.14"
   - Алгоритми: "Сортування бульбашкою: порівнюємо сусідні елементи"
   - Синтаксис: "Оголошення змінної: ключове_слово ім'я = значення"

2. **SOLVE (завдання)** - просить щось зробити, відповісти, створити, застосувати знання
   Приклади (універсальні домени):
   - Питання: "Як мене звати?", "Що таке pi?", "Де столиця?"
   - Завдання: "Створи рецепт салату", "Склади алгоритм сортування"
   - Генерація: "Напиши процедуру для...", "Покажи як використати..."
   - Застосування: "Використовуючи синтаксис що я дав, створи..."
   - Перевірка: "Чи правильно я зробив?", "Перевір цей алгоритм"

**КРИТИЧНО:**
- НЕ припускай що це про конкретну предметну область (код, документи, вірші)
- Класифікуй тільки ЗА ФОРМОЮ: дає інформацію (TEACH) чи просить дію (SOLVE)
- Будь універсальним

Відповідай JSON: {"intent": "teach" або "solve", "confidence": 0.0-1.0, "reasoning": "пояснення"}"""

    try:
        result = await llm.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["message_text"]}
            ],
            response_format=IntentClassification,
            temperature=0.1
        )

        logger.info(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
        logger.info(f"Reasoning: {result.reasoning}")

        return {
            "intent": result.intent,
            "confidence": result.confidence
        }

    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        # Default to SOLVE on error (safer to treat as task)
        return {
            "intent": "solve",
            "confidence": 0.5
        }
