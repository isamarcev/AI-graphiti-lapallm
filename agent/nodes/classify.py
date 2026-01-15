

import logging
from typing import Dict, Any

from pydantic import BaseModel

from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)

@traceable(name="classify_intent")
async def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Classify intent of user message.

    Args:
        state: Current agent state

    Returns:
        State update with intent
    """
    logger.info("=== Classify Intent Node ===")

    llm = get_llm_client()

    system_prompt = """
Класифікуй повідомлення

Визнач чи користувач:

1. **LEARN (навчання)** - дає інформацію, факти, описує, пояснює, навчає
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

Відповідай JSON: {"intent": "learn" або "solve"}"""

    try:
        result = await llm.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["message_text"]}
            ],
            temperature=0.1
        )
        logger.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info(f"Intent classification result: {result}")
        logger.info(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        return {
            "intent": result.intent
        }
    
    except Exception as e:
        logger.error(f"Error classifying intent{e}", exc_info=True)
        return {
            "intent": "learn"
        }
