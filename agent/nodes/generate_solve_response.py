"""
Node: Generate Solve Response
Single unified node that takes context and generates a structured final response.
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)

# --- Схема відповіді (для Структурованого виводу) ---
class AgentResponseSchema(BaseModel):
    reasoning: str = Field(
        description="Внутрішній аналіз (Chain-of-Thought): який формат очікує користувач, які джерела використано, як побудувати відповідь. Це поле НЕ показується користувачу."
    )
    references: List[str] = Field(
        description="Список ТІЛЬКИ значень id з тегів <source id='...'>, БЕЗ тегів та лапок. Приклад: якщо джерело <source id='abc-123'>, то в список додай 'abc-123'. НЕ вигадуй URL чи інші ID."
    )
    response: str = Field(
        description="ФІНАЛЬНА ВІДПОВІДЬ для користувача. Дотримуйся ТОЧНОГО формату, який вказав користувач. Це ЄДИНЕ, що побачить користувач."
    )

@traceable(name="generate_solve_response")
async def generate_solve_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Main node for generating the final answer based on retrieved context.
    Combines 'thinking' (CoT), citation extraction, and formatting in one LLM call.
    """
    logger.info("=== Generate Solve Response Node (Unified) ===")

    message_text = state.get("message_text", "")
    # NOTE: Використовуємо retrieved_context напряму (actualize_context видалено)
    # Fallback на actualized_context для зворотної сумісності
    retrieved_context = state.get("retrieved_context", []) or state.get("actualized_context", [])
    learn_response = state.get("learn_response", "") # Якщо є додаткова інформація з пам'яті

    # 1. Підготовка контексту з явними ID
    if retrieved_context:
        context_parts = []
        for ctx in retrieved_context:
            # Витягуємо ID джерела
            source_id = ctx.get("source_msg_uid") or ctx.get("messageid") or "unknown"
            # Витягуємо контент
            content = ctx.get("content") or ctx.get("fact") or ctx.get("text") or ""

            # Форматуємо як XML-подібний блок для чіткості
            context_parts.append(f"<source id='{source_id}'>\n{content}\n</source>")

        context_text = "\n\n".join(context_parts)
    else:
        context_text = "Контекст відсутній."

    # 2. Універсальний промпт (Без хардкоду домену)
    system_prompt = """Ти — універсальний інтелектуальний агент. Ти генеруєш відповіді ВИКЛЮЧНО на основі наданого контексту.

**ПРОЦЕС РОБОТИ:**
1. Уважно прочитай запит користувача.
2. Визнач ТОЧНИЙ формат, який очікує користувач (вказано в запиті).
3. Знайди потрібну інформацію в блоках `<source id='...'>`.
4. Проаналізуй ЛОГІКУ того, що потрібно зробити.
5. Сформуй відповідь, яка ТОЧНО відповідає запиту за змістом І форматом.

**КРИТИЧНІ ПРАВИЛА:**

1. **ФОРМАТ ВІДПОВІДІ — НАЙВАЖЛИВІШЕ:**
   - Якщо користувач вказав конкретний формат (наприклад, ```щось ... ```) — використай САМЕ його.
   - Якщо просить "без пояснень", "тільки результат" — НЕ додавай вступів, коментарів, пояснень.
   - Відповідь має містити ТІЛЬКИ те, що просив користувач.

2. **ТОЧНІСТЬ ТА СИНТАКСИС:**
   - Використовуй ТІЛЬКИ інформацію з контексту.
   - Не вигадуй нічого, чого немає в джерелах.
   - Якщо в контексті є приклади синтаксису/формату — копіюй їх ТОЧНО.
   - Дотримуйся структури та правил з контексту.

3. **ЛОГІЧНА КОРЕКТНІСТЬ:**
   - Перевір, що твоя відповідь логічно правильна.
   - Якщо потрібно виконати обчислення — результат має бути використаний.
   - Якщо потрібно повернути кілька значень — використай відповідну структуру (список, об'єкт, форматований рядок).
   - Уникай недосяжних частин (наприклад, після повернення результату нічого не виконується).

4. **REFERENCES — ДУЖЕ ВАЖЛИВО:**
   - В поле `references` додавай ТІЛЬКИ значення id з тегів `<source id='...'>`.
   - Приклад: якщо використав `<source id='abc-123-def'>`, додай в references: `abc-123-def`
   - НЕ копіюй весь тег, НЕ вигадуй URL, НЕ додавай нічого іншого.
   - Додавай ТІЛЬКИ ті id, інформацію з яких ти реально використав.

5. **REASONING — ВИКОРИСТОВУЙ ЙОГО:**
   - В полі `reasoning` ОБОВ'ЯЗКОВО напиши свій аналіз:
     * Що саме просить користувач?
     * Який формат відповіді очікується?
     * Які джерела містять потрібну інформацію?
     * Яка логіка рішення?
   - Це поле НЕ показується користувачу, але допомагає тобі думати.

6. **RESPONSE — ЧИСТИЙ РЕЗУЛЬТАТ:**
   - `response` — ЄДИНЕ, що побачить користувач.
   - Жодних зайвих слів, пояснень чи вступів (якщо не просили).

7. **ВІДСУТНІСТЬ ДАНИХ:**
   - Якщо потрібної інформації немає: "На жаль, у мене немає інформації про це."

**ГОЛОВНЕ:** Читай запит користувача ДУЖЕ уважно. Дотримуйся формату БУКВАЛЬНО. Перевіряй логічну коректність.
"""

    user_prompt = f"""**КОНТЕКСТ:**
{context_text}

**ЗАПИТАННЯ КОРИСТУВАЧА:**
{message_text}
"""

    llm_client = get_llm_client()

    try:
        # Використовуємо structured output для гарантованого формату JSON
        result: AgentResponseSchema = await llm_client.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AgentResponseSchema,
            temperature=0.0
        )

        logger.info(f"Generated response. Refs: {result.references}")
        logger.debug(f"Reasoning: {result.reasoning}")

        # Якщо є learn_response (щось, що агент вивчив паралельно), додаємо його
        final_response_text = result.response
        if learn_response:
            # Логіка об'єднання: якщо це код, краще не псувати його текстом,
            # але для загальних відповідей додаємо.
            # Простий варіант:
            if "```" not in final_response_text:
                final_response_text += f"\n\nТакож я запам'ятав: {learn_response}"

        return {
            "response": final_response_text,
            "references": result.references,
            "reasoning": result.reasoning
        }

    except Exception as e:
        logger.error(f"Error in generate_solve_response: {e}", exc_info=True)
        return {
            "response": "Вибачте, сталася технічна помилка при обробці запиту.",
            "references": [],
            "reasoning": f"Exception: {str(e)}"
        }