"""
Node: Generate Solve Response
Extracts the final thought/decision from ReAct steps as the response.
"""

import logging
import re
from typing import Dict, Any, List
from agent.state import AgentState
from clients.llm_client import get_llm_client
from langsmith import traceable
from config.settings import settings

logger = logging.getLogger(__name__)


def extract_references(text: str) -> List[str]:
    """Extract references from text matching [джерело: X] pattern."""
    # Match patterns like [джерело: 1], [джерело: abc123], etc.
    pattern = r'\[джерело:\s*([^\]]+)\]'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Deduplicate while preserving order
    seen = set()
    references = []
    for ref in matches:
        ref = ref.strip()
        if ref and ref not in seen:
            seen.add(ref)
            references.append(ref)
    
    return references


@traceable(name="generate_solve_response")
async def generate_solve_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Генерує відповідь для SOLVE режиму, використовуючи LLM для витягування прямої відповіді.

    Args:
        state: Current agent state з solve_response та message_text

    Returns:
        State update with:
        - response: extracted direct answer
        - references: extracted from [джерело: X] patterns
        - reasoning: empty string
    """
    logger.info("=== Generate Solve Response Node ===")

    message_text = state.get("message_text", "")
    solve_response = state.get("solve_response", "")
    learn_response = state.get("learn_response", "")
    
    if not solve_response:
        logger.warning("No solve_response available")
        return {
            "response": learn_response if learn_response else "Не вдалося згенерувати відповідь",
            "references": [],
            "reasoning": ""
        }
    
    # Extract references from original solve_response
    references = extract_references(solve_response)
    logger.info(f"Extracted {len(references)} references: {references}")
    
    # Use LLM to extract direct answer
    llm_client = get_llm_client()
    
    system_prompt = """Ти експерт з витягування прямих відповідей на запитання.

**ТВОЯ ЗАДАЧА:**
З наданого тексту відповіді витягни ТІЛЬКИ пряму, коротку відповідь на конкретне запитання користувача.

**ПРАВИЛА:**
1. Відповідай ТІЛЬКИ на запитання - не додавай зайвої інформації
3. Зберігай всі посилання на джерела у форматі [джерело: X]
4. Якщо у тексті є кілька тем - витягни ТІЛЬКИ ту, що стосується запитання
5. Відповідай українською мовою

**ПРИКЛАДИ:**

Запитання: "Яка столиця України?"
Текст: "Україна - велика країна. Столиця України - Київ [джерело: 1]. Київ розташований на Дніпрі. Населення міста понад 3 мільйони."
Відповідь: "Столиця України - Київ [джерело: 1]."

Запитання: "Що любить їсти Марія?"
Текст: "Марія - вчителька математики [джерело: 1]. Вона працює в школі №5. Марія обожнює борщ і вареники [джерело: 2]. У вільний час читає книги."
Відповідь: "Марія обожнює борщ і вареники [джерело: 2]."
"""

    user_prompt = f"""**ЗАПИТАННЯ КОРИСТУВАЧА:**
{message_text}

**ТЕКСТ ВІДПОВІДІ:**
{solve_response}

**ПРЯМА ВІДПОВІДЬ:**"""

    try:
        extracted_answer = await llm_client.generate_async(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=settings.temperature,
            max_tokens=500
        )
        
        logger.info(f"Extracted direct answer: {extracted_answer[:100]}...")
        
        # Combine with learn_response if present
        final_response = extracted_answer
        if learn_response:
            final_response = extracted_answer + "\n\n" + learn_response
        
        return {
            "response": final_response,
            "references": references,
            "reasoning": ""
        }
        
    except Exception as e:
        logger.error(f"Error extracting direct answer: {e}", exc_info=True)
        # Fallback to original behavior
        response = solve_response + "\n\n" + learn_response if learn_response else solve_response
        return {
            "response": response,
            "references": references,
            "reasoning": ""
        }