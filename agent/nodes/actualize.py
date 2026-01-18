"""
Node for context actualization - extracts only relevant context items based on task and plan.
Uses brief_fact for LLM analysis, returns full facts for relevant items.
"""

import json
import logging
from typing import Dict, Any, List
from langsmith import traceable
from pydantic import BaseModel

from agent.state import AgentState
from clients.llm_client import get_llm_client
from config.settings import settings

logger = logging.getLogger(__name__)


class RelevanceAnalysis(BaseModel):
    """Structured output for relevance analysis."""
    relevant_indexes: List[int]


@traceable(name="actualize_context")
async def actualize_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyze retrieved context and extract only relevant items based on task and plan.
    
    Uses brief_fact field for LLM analysis to determine relevance.
    Returns list of relevant items with full fact and message_id (no brief_fact).
    
    Args:
        state: AgentState with message_text, plan, and retrieved_context
        
    Returns:
        State update with relevant_context list
    """
    logger.info("=== Actualize Context Node ===")
    
    # Extract data from state
    message_text = state.get("message_text", "")
    plan = state.get("plan", "")
    retrieved_context = state.get("retrieved_context", [])
    
    # Early exit: no context to analyze
    if not retrieved_context:
        logger.info("No context to analyze, returning empty")
        return {"relevant_context": []}
    
    original_count = len(retrieved_context)
    logger.info(f"Analyzing {original_count} context items for relevance")
    
    # Build brief facts list for LLM
    brief_facts_list = []
    for i, ctx in enumerate(retrieved_context):
        brief_fact = ctx.get("brief_fact", "") or ctx.get("content", "")[:100]
        brief_facts_list.append(f"{i}. {brief_fact}")
    
    brief_facts_string = "\n".join(brief_facts_list)
    
    # Build prompt for LLM
    system_prompt = """Ти експерт з аналізу релевантності інформації для виконання задач. Тобі надана задача, план її виконання та список назв фактів потенційно релевантних для виконання задачі.

**ТВОЯ ЗАДАЧА:**
Проаналізуй задачу користувача і план її виконання. Визнач за назвою, які факти ДІЙСНО релевантні для виконання поставленої задачі.

**КРИТЕРІЇ РЕЛЕВАНТНОСТІ:**
✓ Теоретичні факти що необхідні для виконання задачі
✓ Інформація, що безпосередньо стосується запиту чи якогось з пунктів плану

**КРИТЕРІЇ НЕРЕЛЕВАНТНОСТІ:**
✗ Загальна інформація, яка не стосується даної задачі фбо конкретного пункту плану
✗ Інформація про інші теми/алгоритми/об'єкти
✗ Побічна інформація, яка точно не допоможе виконати задачу

**ФОРМАТ ВІДПОВІДІ:**
Поверни JSON з масивом індексів релевантних фактів:
{"relevant_indexes": [...]}

Якщо жоден факт не релевантний, поверни порожній масив: {"relevant_indexes": []}"""

    user_prompt = f"""Задача: {message_text}

План: {plan if plan else "(план відсутній)"}

Факти для аналізу:
{brief_facts_string}"""

    try:
        # Call LLM to identify relevant items
        llm_client = get_llm_client()
        
        # Try structured output first
        try:
            result = await llm_client.generate_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format=RelevanceAnalysis
            )
            relevant_indexes = result.relevant_indexes
        except Exception as e:
            logger.warning(f"Structured output failed, trying JSON parsing: {e}")
            # Fallback to JSON parsing
            response = await llm_client.generate_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            parsed = json.loads(response)
            relevant_indexes = parsed.get("relevant_indexes", [])
        
        logger.info(f"LLM identified {len(relevant_indexes)} relevant items: {relevant_indexes}")
        
        # Build relevant context list with full facts
        relevant_context = []
        for idx in relevant_indexes:
            if 0 <= idx < len(retrieved_context):
                ctx = retrieved_context[idx]
                relevant_context.append({
                    "content": ctx.get("content", ""),
                    "message_id": ctx.get("message_id", "unknown")
                })
            else:
                logger.warning(f"Invalid index {idx} from LLM (max: {len(retrieved_context)-1})")
        
        logger.info(f"Filtered context: {len(relevant_context)}/{original_count} items")
        
        return {"relevant_context": relevant_context}
        
    except Exception as e:
        logger.error(f"Error during context actualization: {e}", exc_info=True)
        logger.warning("Falling back to all context due to error")
        
        # Graceful fallback: return all context without brief_fact
        fallback_context = []
        for ctx in retrieved_context:
            fallback_context.append({
                "content": ctx.get("content", ""),
                "message_id": ctx.get("message_id", "unknown")
            })
        
        return {"relevant_context": fallback_context}
