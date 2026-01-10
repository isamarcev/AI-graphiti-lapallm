"""
Node 8: Answer Generation
Генерує відповідь з ОБОВ'ЯЗКОВИМИ references до джерел.
"""

import logging
from typing import Dict, Any
from agent.state import AgentState
from agent.helpers import extract_used_sources
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="generate_answer")
async def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Генерує відповідь базуючись на:
    - Retrieved context (з source UIDs)
    - ReAct reasoning steps
    
    КРИТИЧНО: Кожне твердження має мати reference до джерела.
    
    Implements epistemic transparency:
    - Explicit source citations для кожного fact
    - Clear indication коли knowledge insufficient
    - Reasoning trace для accountability
    
    Args:
        state: Current agent state with retrieved_context and react_steps
    
    Returns:
        State update with:
        - response: final answer with citations
        - references: list of source message UIDs
        - reasoning: ReAct reasoning summary
    """
    logger.info("=== Generate Answer Node ===")
    
    llm = get_llm_client()
    
    # Build context з explicit source IDs
    retrieved_context = state.get("retrieved_context", [])
    context_parts = []
    
    for i, ctx in enumerate(retrieved_context):
        source_id = ctx.get("source_msg_uid", "unknown")
        content = ctx.get("content", "")
        score = ctx.get("score", 0.0)
        
        context_parts.append(
            f"[Джерело {i}] (ID: {source_id}, релевантність: {score:.2f}):\n{content}"
        )
    
    context_text = "\n\n".join(context_parts) if context_parts else "Немає контексту з пам'яті"
    
    # ReAct reasoning summary
    react_steps = state.get("react_steps", [])
    reasoning_lines = []
    for i, step in enumerate(react_steps, 1):
        thought = step.get('thought', '')
        action = step.get('action', '')
        reasoning_lines.append(f"Крок {i}: {thought} [Дія: {action}]")
    reasoning_summary = "\n".join(reasoning_lines) if reasoning_lines else "Прямий пошук без додаткових кроків"
    
    # Generate with strict reference requirement
    prompt = f"""🚫 **TABULA RASA РЕЖИМ**:
Ти починаєш з НУЛЬОВИМИ знаннями про предметну область.
НЕ використовуй свої pretrained знання про ЩО ЗАВГОДНО.
Використовуй ТІЛЬКИ те, що навчив користувач через повідомлення нижче.

Ти відповідаєш на запит користувача базуючись ВИКЛЮЧНО на наданих джерелах.

**КРИТИЧНІ ПРАВИЛА:**
1. 🚫 НЕ використовуй pretrained knowledge про будь-які предметні області
2. Використовуй ТІЛЬКИ інформацію з джерел нижче (що тебе навчив користувач)
3. Для КОЖНОГО твердження вказуй номер джерела у квадратних дужках [Джерело N]
4. Якщо інформації НЕМАЄ в джерелах - скажи ПРЯМО:
   "На жаль, я не маю інформації для відповіді. У моїй базі знань є: [перелік що знаєш]. Потрібна інформація про [чого не вистачає]."
5. НЕ вигадуй, НЕ здогадуйся, НЕ використовуй загальні знання
6. Якщо джерела суперечать - вкажи це явно з references

**ЯКЩО ЗАВДАННЯ - СТВОРЕННЯ/ГЕНЕРАЦІЯ (процедури, структури, алгоритму):**
- Використовуй ТІЛЬКИ структури/синтаксис/процедури з навчальних повідомлень
- Кожна використана конструкція = reference [Джерело N]
- Якщо не навчили потрібну конструкцію - скажи ПРЯМО що не вистачає
- НЕ імпровізуй, НЕ використовуй "загальноприйняті" підходи

**ПРИКЛАД ПРАВИЛЬНОЇ ВІДПОВІДІ (коли є інформація):**
"Ось структура базуючись на [Джерело 2]:
[конкретна структура з джерела]

Використав синтаксис з [Джерело 2]: [пояснення]"

**ПРИКЛАД КОЛИ НЕ НАВЧЕНИЙ:**
"На жаль, я ще не навчений як робити X. У моїй базі знань є:
- [факт 1] [Джерело 3]
- [факт 2] [Джерело 5]

Потрібна інформація про [конкретно чого не вистачає]."

КОНТЕКСТ З ПАМ'ЯТІ (що тебе навчили):
{context_text}

ПРОЦЕС МІРКУВАННЯ (ReAct):
{reasoning_summary}

ЗАПИТ КОРИСТУВАЧА: {state['message_text']}

Відповідай українською мовою. Обов'язково вказуй джерела у форматі [Джерело N]."""
    
    try:
        response_text = await llm.generate_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        
        logger.debug(f"Generated response: {response_text[:200]}...")
        
        # Extract використані джерела з тексту
        used_source_indices = extract_used_sources(response_text)
        
        # Map indices to actual message UIDs
        used_sources = set()
        for idx in used_source_indices:
            if idx < len(retrieved_context):
                source_uid = retrieved_context[idx].get("source_msg_uid")
                if source_uid and source_uid != "unknown":
                    used_sources.add(source_uid)
                    logger.debug(f"Using source {idx}: {source_uid}")
        
        # Додати current message UID як reference
        references = list(used_sources)
        if state["message_uid"] not in references:
            references.append(state["message_uid"])
        
        logger.info(f"Generated answer with {len(references)} references")
        
        # Check if response indicates insufficient knowledge
        insufficient_indicators = [
            "не маю",
            "недостатньо інформації",
            "немає інформації",
            "не знайдено"
        ]
        if any(indicator in response_text.lower() for indicator in insufficient_indicators):
            logger.warning("Agent indicates insufficient knowledge to answer")
        
        return {
            "response": response_text,
            "references": references,
            "reasoning": reasoning_summary
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return {
            "response": f"Помилка при генерації відповіді: {str(e)}",
            "references": [state["message_uid"]],
            "reasoning": None
        }
