"""
Simple Context-Based Answer Node.
Takes retrieved context and user query, returns answer based on context only.
No tools, no ReAct loop - just a single LLM call.
"""

import logging
from typing import Any, Dict

from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable
from config.settings import settings

logger = logging.getLogger(__name__)


@traceable(name="context_answer")
async def context_answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Simple node that answers user query based on retrieved context.
    
    Args:
        state: AgentState with retrieved_context and message_text
        
    Returns:
        State update with response
    """
    logger.info("=== Context Answer Node ===")
    
    # Get inputs
    message_text = state.get("message_text", "")
    relevant_context_list = state.get("relevant_context", [])
    plan = state.get("plan", "")
    
    # Format relevant_context list into string
    if not relevant_context_list:
        context_string = "(контекст порожній)"
    else:
        context_parts = []
        for i, ctx in enumerate(relevant_context_list, 1):
            content = ctx.get("content", "")
            message_id = ctx.get("message_id", "unknown")
            context_parts.append(f"{i}. {content}\n[джерело: {message_id}]")
        context_string = "\n\n".join(context_parts)
    
    logger.info(f"Formatted {len(relevant_context_list)} context items")

    system_prompt = """Ти асистент, який відповідає ТІЛЬКИ на основі наданого контексту.

🚫 ЗАБОРОНЕНО використовувати будь-які знання поза контекстом.
✅ Відповідай ТІЛЬКИ якщо інформація є в контексті.

ПРАВИЛА:
1. Якщо відповідь є в контексті → дай відповідь
2. Якщо відповіді НЕМАЄ в контексті → скажи "Не маю інформації про це"
3. ОБОВ'ЯЗКОВО вказуй джерела у форматі [джерело: X]
4. Відповідай українською мовою"""

    user_prompt = f"""КОНТЕКСТ:
{context_string}

ЗАПИТАННЯ: {message_text}

Орієнтовний план виконання:
{plan}

ВІДПОВІДЬ:"""

    # Call LLM
    llm_client = get_llm_client()
    
    try:
        response = await llm_client.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=settings.temperature
        )
        
        logger.info(f"Generated response: {response[:100]}...")
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        response = "Помилка генерації відповіді"
    
    return {
        "solve_response": response
    }
