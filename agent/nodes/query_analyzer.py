"""
Query Analyzer Node - аналізує запит користувача та формує стратегію пошуку.

Цей вузол розуміє ЩО потрібно для відповіді та генерує оптимізовані пошукові запити.
"""

import logging
import json
from typing import Dict, Any, List
from langsmith import traceable

from agent.state import AgentState
from clients.llm_client import get_llm_client
from models.schemas import PlanAnalysis
from config.settings import settings

logger = logging.getLogger(__name__)


def _build_analysis_prompt(message_text: str) -> list[dict[str, str]]:
    """
    Формує промпт для аналізу запиту користувача.

    Args:
        message_text: Запит користувача

    Returns:
        Список повідомлень для LLM
    """
    system_prompt = """Ти експерт-планувальник. Твоя задача: уважно проаналізувати задачу користувача, створити детальний і вичерпний план для вирішення задачі. Також ти маєш визначити яка інформація необхідна для вирішення пунктів плану та задачі і в цілому. 
    Створи докладний покроковий план виконання задачі . Після цього детально опиши яка інформація тобі потрібна для точного розв'язання задачі. Відповідь має містити план та перелік конкретних нюансів необхідної теорії якої бракує для виконання пунктів плану та задачі. 
    """

    user_prompt = f"""**ЗАПИТ КОРИСТУВАЧА:**
{message_text}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _build_search_queries_prompt(plan: str):
    system_prompt = """Ти - експерт з формування запитів для векторного пошуку необхідної інформації. Твоя задача проаналізувати наданий план та свормувати список запитів для пошуку в знаннях. 
    Уважно проаналізуй план та інформацію якої бракує для вирішення задачі. Сформуй список точних запитів для пошуку в знаннях. Запит не має бути загальним, а повинен бути максимально точним та конкретним.

    Правила формування запитів:
    1. Запит не має бути загальним, це має бути запит на конкретний шматок інформації.
    2. Запит повинен бути максимально точним та конкретним та стосуватися певної деталі.

    Приклади запитів:
    НЕПРАВИЛЬНО: "Права людини"
    ПРАВИЛЬНО: "Права людини щодо власного майна у сучасному світі"

    НЕПРАВИЛЬНО: "Респіраторні захворення"
    ПРАВИЛЬНО: "Опис первинних симптомів респіраторних захворювань"

    НЕПРАВИЛЬНО: "Мова програмування JavaScript"
    ПРАВИЛЬНО: "Цикли в JavaScript"


    Відповідай в форматі JSON ["запит1", "запит2", ...]"""
    user_prompt = f"План: {plan}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _parse_search_queries(raw_response: str, fallback_query: str) -> List[str]:
    """
    Parse LLM response to extract valid JSON list of search queries.
    
    Args:
        raw_response: Raw LLM response that should contain JSON array
        fallback_query: Fallback query to use if parsing fails
        
    Returns:
        List of search query strings
    """
    try:
        # Clean markdown code blocks if present
        response_text = raw_response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        # Validate it's a list
        if not isinstance(parsed, list):
            logger.warning(f"Parsed result is not a list: {type(parsed)}")
            return [fallback_query]
        
        # Validate all items are strings and non-empty
        queries = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())
            else:
                logger.warning(f"Skipping invalid query item: {item}")
        
        # Ensure we have at least one query
        if not queries:
            logger.warning("No valid queries found in parsed result")
            return [fallback_query]
        
        # Limit to max 3 queries
        if len(queries) > 3:
            logger.info(f"Limiting {len(queries)} queries to top 3")
            queries = queries[:3]
        
        return queries
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from search queries response: {e}")
        logger.debug(f"Raw response: {raw_response[:200]}")
        return [fallback_query]
    except Exception as e:
        logger.error(f"Unexpected error parsing search queries: {e}", exc_info=True)
        return [fallback_query]
@traceable(name="query_analyzer")
async def query_analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Аналізує запит користувача та формує стратегію пошуку контексту.

    Цей вузол:
    1. Виділяє ключові сутності
    2. Формує чек-лист необхідної інформації для повної відповіді
    3. Генерує оптимізовані пошукові запити

    Args:
        state: AgentState з message_text

    Returns:
        State update з query_analysis
    """
    logger.info("=== Query Analyzer Node ===")

    message_text = state.get("message_text", "")

    if not message_text:
        logger.warning("Empty message_text, skipping analysis")
        # Fallback: створюємо базовий аналіз
        return {
            "plan": "Знайти релевантну інформацію та сформулювати відповідь",
            "search_queries": [message_text] if message_text else []
        }

    logger.info(f"Analyzing query: '{message_text}'")

    try:
        # Формуємо промпт
        analysis_messages = _build_analysis_prompt(message_text)

        # Викликаємо LLM зі structured output для отримання плану та необхідної інформації
        llm_client = get_llm_client()
        plan_analysis: PlanAnalysis = await llm_client.generate_async(
            messages=analysis_messages,
            temperature=settings.temperature,
            max_tokens=500,
            response_format=PlanAnalysis
        )

        # Формуємо текст для генерації пошукових запитів
        plan_text = f"План: {plan_analysis.plan}\n\nНеобхідна інформація: {plan_analysis.required_info}"
        search_queries_messages = _build_search_queries_prompt(plan_text)
        search_queries_raw = await llm_client.generate_async(
            messages=search_queries_messages,
            temperature=settings.temperature,
            max_tokens=500
        )

        # Parse and validate search_queries as JSON list
        search_queries = _parse_search_queries(search_queries_raw, message_text)
        
        logger.info(f"Generated plan: {plan_analysis.plan[:100]}...")
        logger.info(f"Required info: {plan_analysis.required_info[:100]}...")
        logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")

        return {"plan": plan_analysis.plan, "search_queries": search_queries}

    except Exception as e:
        logger.error(f"Error during query analysis: {e}", exc_info=True)
        logger.warning("Falling back to basic analysis")

        # Graceful fallback: використовуємо оригінальний запит
        return {
            "plan": "Знайти релевантну інформацію та сформулювати відповідь",
            "search_queries": [message_text]
        }
