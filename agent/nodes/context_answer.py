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
    retrieved_context = state.get("actualized_context", [])

    # Format context
    if retrieved_context:
        context_parts = []
        for i, ctx in enumerate(retrieved_context, 1):
            content = ctx.get("content", "") or ctx.get("fact", "")
            description = ctx.get("description", "")
            examples = ctx.get("examples", "")
            source = ctx.get("source_msg_uid") or ctx.get("messageid") or "unknown"
            
            part = f"{i}. Факт: {content}"
            if description:
                part += f"\n   Опис: {description}"
            if examples:
                # examples can be string or list
                if isinstance(examples, list):
                    part += f"\n   Приклади: {', '.join(examples)}"
                else:
                    part += f"\n   Приклади: {examples}"
            part += f"\n   [джерело: {source}]"
            context_parts.append(part)
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "(контекст порожній)"
    logger.info("*" * 50)
    logger.info(f"Retrieved context: {context_text}")
    logger.info("*" * 50)
    # Build prompt
    system_prompt = """Ти асистент, який відповідає ТІЛЬКИ на основі наданого контексту.

🚫 ЗАБОРОНЕНО використовувати будь-які знання поза контекстом.
✅ Відповідай ТІЛЬКИ якщо інформація є в контексті.

ПРАВИЛА:
1. Якщо відповідь є в контексті → дай коротку відповідь (2-4 речення)
2. Якщо відповіді НЕМАЄ в контексті → скажи "Не маю інформації про це"
3. ОБОВ'ЯЗКОВО вказуй джерела у форматі [джерело: X]
4. Відповідай українською мовою"""

    user_prompt = f"""КОНТЕКСТ:
{context_text}

ЗАПИТАННЯ: {message_text}

ВІДПОВІДЬ:"""

    # Call LLM
    llm_client = get_llm_client()
    
    try:
        response = await llm_client.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.001
        )
        
        logger.info(f"Generated response: {response[:100]}...")
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        response = "Помилка генерації відповіді"
    
    return {
        "solve_response": response
    }
