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
        description="Короткий опис (Chain-of-Thought): аналіз запиту, які фрагменти обрано, чому саме вони, і як формується відповідь."
    )
    references: List[str] = Field(
        description="Список ID джерел (source_uid), інформація з яких була використана. Наприклад: ['msg_123', 'doc_5']"
    )
    response: str = Field(
        description="Фінальна відповідь для користувача. Має точно відповідати стилю/формату, який просив користувач (наприклад, тільки код)."
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
    system_prompt = """Ти — інтелектуальний агент, який генерує точні відповіді виключно на основі наданого контексту.

**ТВОЇ ЗАВДАННЯ:**
1. **Аналіз (Reasoning):** Прочитай запит та знайди відповідну інформацію в блоках `<source id='...'>`. Проаналізуй, який формат відповіді очікує користувач.
2. **Цитування (References):** Збери ID всіх джерел, які ти використав для відповіді.
3. **Генерація (Response):** Сформуй фінальну відповідь українською мовою.

**ВАЖЛИВІ ПРАВИЛА ДЛЯ ПОЛЯ `response`:**
- **Точність:** Використовуй ТІЛЬКИ факти з контексту. Не вигадуй.
- **Формат:** Якщо користувач просить "тільки код", "JSON", "одним словом" — суворо дотримуйся цього. Не пиши вступів типу "Ось ваш код".
- **Відсутність даних:** Якщо інформації немає в контексті, в полі `response` напиши: "На жаль, у мене немає інформації про це в доступних джерелах."
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