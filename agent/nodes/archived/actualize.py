"""
Node: Context Actualization (Rerank & Filter)
Filters retrieved context items by relevance using an LLM as a Judge.
Prevents context pollution and reduces token usage for the final generation.
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langsmith import traceable

from agent.state import AgentState
from clients.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Output ---

class RelevanceVerdict(BaseModel):
    index: int = Field(..., description="Індекс елемента контексту (починаючи з 0)")
    is_relevant: bool = Field(..., description="True, якщо цей фрагмент корисний для відповіді")
    reasoning: str = Field(..., description="Дуже коротке пояснення (3-5 слів), чому це релевантно чи ні")

class ContextEvaluation(BaseModel):
    verdicts: List[RelevanceVerdict] = Field(..., description="Оцінка для кожного елемента контексту")

# --- Helper Functions ---

def _format_context_for_judge(context_items: List[Dict[str, Any]]) -> str:
    """Creates a numbered list of context items for the LLM to judge."""
    formatted = []
    for i, item in enumerate(context_items):
        # Retrieve node returns "content", old nodes might use "fact"
        content = item.get("content") or item.get("fact", "")
        # Обрізаємо дуже довгі шматки для економії на етапі оцінки
        preview = content
        formatted.append(f"ITEM [{i}]: {preview}")
    return "\n\n".join(formatted)

# --- Node Implementation ---

@traceable(name="actualize_context")
async def actualize_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Filters retrieved context to keep only relevant items.
    Acts as a Reranker/Filter step between Retrieve and Generate.
    """
    logger.info("=== Actualize Context Node (Smart Filter) ===")

    # 1. Отримуємо вхідні дані
    message_text = state.get("message_text", "")
    retrieved_context = state.get("retrieved_context", [])

    # Якщо нічого не знайшли раніше — немає чого фільтрувати
    if not retrieved_context:
        logger.warning("No context to filter.")
        return {"actualized_context": []}

    # Якщо контексту мало (наприклад, 1-2 елементи), фільтрація може бути зайвою
    # Але для чистоти експерименту можна залишити завжди.

    context_str = _format_context_for_judge(retrieved_context)

    # 2. Формуємо промпт "Суворого Судді"
    system_prompt = """Ти — аналітик-фільтр для RAG системи. Твоє завдання — відсіяти інформаційне сміття.

Тобі надано запит користувача та список знайдених фрагментів тексту.
Для кожного фрагмента визнач, чи він допомагає відповісти на запит.

**КРИТЕРІЇ РЕЛЕВАНТНОСТІ:**
✅ **KEEP (True):**
- Містить пряму відповідь або її частину.
- Містить важливі визначення, факти чи цифри, синтаксис чи правила, що стосуються теми.
- Містить контекст, необхідний для розуміння (наприклад, про сутність, згадану в запиті).

❌ **DISCARD (False):**
- Інформація про зовсім інші речі (омоніми, інші люди з таким ім'ям).
- Загальні фрази, "вода", метадані без змісту.
- Інформація, що суперечить темі запиту (якщо запит про "яблука", а текст про "груші", і це не порівняння).

**ФОРМАТ:**
Поверни JSON з оцінкою для КОЖНОГО фрагмента.
"""

    user_prompt = f"""**ЗАПИТ КОРИСТУВАЧА:**
"{message_text}"

**КАНДИДАТИ НА КОНТЕКСТ:**
{context_str}

Оціни кожен ITEM. Будь критичним, але не викидай корисну інформацію."""

    llm_client = get_llm_client()

    try:
        # 3. Виклик LLM (Structured Output)
        # Використовуємо temperature=0 для стабільності
        evaluation: ContextEvaluation = await llm_client.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ContextEvaluation,
            temperature=0.0
        )

        # 4. Фільтрація
        filtered_context = []
        # Створюємо мапу вердиктів для швидкого доступу
        verdict_map = {v.index: v for v in evaluation.verdicts}

        for i, item in enumerate(retrieved_context):
            verdict = verdict_map.get(i)

            # Якщо LLM сказала "True" — залишаємо
            if verdict and verdict.is_relevant:
                # Можна додати пояснення в лог для дебагу
                logger.debug(f"Keep Item {i}: {verdict.reasoning}")
                filtered_context.append(item)
            else:
                reason = verdict.reasoning if verdict else "No verdict"
                logger.debug(f"Drop Item {i}: {reason}")

        # 5. SAFETY FALLBACK (Дуже важливо!)
        # Якщо LLM вирішила, що все сміття (фільтр занадто агресивний),
        # а ми знаємо, що векторний пошук щось знайшов — краще повернути хоча б Топ-1,
        # ніж пустий контекст.
        if not filtered_context and retrieved_context:
            logger.warning("⚠️ All items were filtered out! Falling back to Top-2 vector results.")
            # Сортуємо за score (якщо він є) або просто беремо перші
            sorted_context = sorted(retrieved_context, key=lambda x: x.get("score", 0), reverse=True)
            filtered_context = sorted_context[:2]

        logger.info(f"Actualization complete: {len(retrieved_context)} -> {len(filtered_context)} items kept.")

        return {"actualized_context": filtered_context}

    except Exception as e:
        logger.error(f"Error in Actualize Node: {e}", exc_info=True)
        # У разі помилки — не ламаємо пайплайн, віддаємо все, що є
        return {"actualized_context": retrieved_context}