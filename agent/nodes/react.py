"""
Node 7: ReAct Loop
Implements ReAct (Reasoning + Acting) for complex task solving.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Set

from pydantic import BaseModel

from agent.helpers import format_search_results
from agent.state import AgentState
from clients.llm_client import get_llm_client
from clients.qdrant_client import QdrantClient
from clients.hosted_embedder import get_embedder
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)

ACTION_ANSWER = "answer"
ACTION_SEARCH = "search"
ALLOWED_ACTIONS = {ACTION_ANSWER, ACTION_SEARCH}
ANSWER_KEYWORDS = ("готовий", "достатньо", "можу відповісти", "answer", "ready")
SEARCH_KEYWORDS = ("шукати", "знайти", "потрібно", "search", "find", "need")

MIN_QUERY_LENGTH = 3  # Minimum query length in characters


class ReactStep(BaseModel):
    thought: str
    action: str
    query: Optional[str] = None


def _build_context_text(retrieved_context: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"[{i}] ({ctx.get('source_msg_uid', 'unknown')}): {ctx.get('content', '')}"
        for i, ctx in enumerate(retrieved_context)
    )


def _build_history_text(steps: List[Dict[str, Any]]) -> str:
    return "\n".join(
        "Крок {idx}:\n  Думка: {thought}\n  Дія: {action}\n  Результат: {observation}".format(
            idx=i + 1,
            thought=step.get("thought", ""),
            action=step.get("action", ""),
            observation=(step.get("observation", "")[:200] + "...") if step.get("observation") else "",
        )
        for i, step in enumerate(steps)
    )


def _build_thought_prompt(task: str, context_text: str, history_text: Optional[str], iteration: int) -> str:
    if iteration == 0:
        return f"""🚫 TABULA RASA: У тебе НУЛЬОВІ знання про предметну область.
Використовуй ТІЛЬКИ інформацію з контексту нижче. НЕ використовуй pretrained knowledge.

Контекст з пам'яті (що тебе навчили):
{context_text if context_text else "(порожньо - нічого не навчили)"}

Завдання: {task}

ФОРМАТ ВІДПОВІДІ - JSON з ключами:
  "thought": короткий виклад твоєї логіки (1-2 речення)
  "action": "answer" або "search"
  "query": конкретний пошуковий запит (ОБОВ'ЯЗКОВО якщо action="search")

ПРАВИЛА:
- Якщо в контексті Є достатня інформація → action="answer", query можна не вказувати
- Якщо в контексті НЕМАЄ потрібної інформації → action="search", query ОБОВ'ЯЗКОВИЙ
- query має бути конкретним (2-5 ключових слів), НЕ повним реченням
- query має відображати ЩО саме шукати, а не "потрібно знайти..."

ПРИКЛАДИ:
{{"thought": "В контексті немає інформації про столицю", "action": "search", "query": "столиця України"}}
{{"thought": "Контекст містить відповідь про Київ", "action": "answer"}}
{{"thought": "Треба дізнатись про улюблену їжу", "action": "search", "query": "улюблена їжа Олега"}}

JSON відповідь:"""

    return f"""🚫 TABULA RASA: Використовуй ТІЛЬКИ інформацію з контексту. НЕ використовуй pretrained knowledge.

Попередні кроки:
{history_text or "(немає)"}

Поточний контекст (що тебе навчили):
{context_text}

Завдання: {task}

ФОРМАТ ВІДПОВІДІ - JSON з ключами:
  "thought": короткий виклад твоєї логіки (1-2 речення)
  "action": "answer" або "search"
  "query": конкретний пошуковий запит (ОБОВ'ЯЗКОВО якщо action="search")

ПРАВИЛА:
- Якщо в контексті Є достатня інформація → action="answer"
- Якщо в контексті НЕМАЄ потрібної інформації → action="search" + query ОБОВ'ЯЗКОВИЙ
- query має бути конкретним (2-5 ключових слів), НЕ повним реченням
- НЕ повторюй попередні запити, шукай щось нове

JSON відповідь:"""


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    # Try direct JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within text (incl. fenced blocks)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _normalize_action(action: str) -> str:
    action_lower = action.strip().lower()
    if action_lower in ALLOWED_ACTIONS:
        return action_lower
    return ACTION_ANSWER


def _infer_action_from_thought(thought: str) -> str:
    thought_lower = thought.lower()
    if any(keyword in thought_lower for keyword in ANSWER_KEYWORDS):
        return ACTION_ANSWER
    if any(keyword in thought_lower for keyword in SEARCH_KEYWORDS):
        return ACTION_SEARCH
    return ACTION_ANSWER


def _parse_react_response(response_text: str) -> Dict[str, Any]:
    payload = _extract_json_payload(response_text)
    if payload:
        thought = str(payload.get("thought", "")).strip()
        action = _normalize_action(str(payload.get("action", "")))
        query = str(payload.get("query", "")).strip()
        return {"thought": thought, "action": action, "query": query}

    # Fallback: treat entire output as thought and infer action
    thought = response_text.strip()
    action = _infer_action_from_thought(thought)
    return {"thought": thought, "action": action, "query": ""}


def _is_valid_query(query: Optional[str]) -> bool:
    """
    Validate if search query is meaningful.
    
    Args:
        query: Search query string
        
    Returns:
        True if query is valid for search
    """
    if not query or not query.strip():
        return False
    
    query_clean = query.strip()
    
    # Too short
    if len(query_clean) < MIN_QUERY_LENGTH:
        return False
    
    # Only punctuation or whitespace
    if not re.search(r'[а-яА-ЯіІїЇєЄa-zA-Z0-9]', query_clean):
        return False
    
    return True


def _validate_react_step(thought: str, action: str, query: Optional[str]) -> tuple[bool, str]:
    """
    Validate ReAct step output from LLM.
    
    Args:
        thought: Agent's reasoning
        action: Action to take
        query: Search query (required if action is search)
        
    Returns:
        (is_valid, error_message) tuple
    """
    if not thought:
        return False, "Thought is empty"
    
    if action not in ALLOWED_ACTIONS:
        return False, f"Invalid action: {action}"
    
    # If action is search, query must be valid
    if action == ACTION_SEARCH:
        if not _is_valid_query(query):
            return False, f"Search action requires valid query, got: '{query}'"
    
    return True, ""


@traceable(name="react_loop")
async def react_loop_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 7: ReAct (Reasoning + Acting) loop.

    Iteratively:
    1. THOUGHT - what needs to be done
    2. ACTION - search for more context OR generate answer
    3. OBSERVATION - result of action

    Continues until:
    - Max iterations reached
    - Action is "answer" (ready to respond)

    Args:
        state: Current agent state

    Returns:
        State update with react_steps
    """
    logger.info("=== ReAct Loop Node ===")

    max_iterations = getattr(settings, 'max_react_iterations', 3)
    logger.info(f"Max iterations: {max_iterations}")

    llm = get_llm_client()
    embedder = get_embedder()
    qdrant = QdrantClient()
    await qdrant.initialize()

    # Build initial context
    retrieved_context = state.get("retrieved_context", [])
    context_text = _build_context_text(retrieved_context)

    task = state["message_text"]
    steps: List[Dict[str, Any]] = []
    searched_queries: Set[str] = set()  # Track queries to avoid duplicates

    for iteration in range(max_iterations):
        logger.info(f"\n--- ReAct Iteration {iteration + 1}/{max_iterations} ---")

        history_text = _build_history_text(steps) if steps else None
        thought_prompt = _build_thought_prompt(task, context_text, history_text, iteration)

        # Generate thought with structured output
        thought = ""
        action = ACTION_ANSWER
        search_query = ""
        
        try:
            structured = await llm.generate_async(
                messages=[{"role": "user", "content": thought_prompt}],
                temperature=0.3,
                max_tokens=200,
                response_format=ReactStep
            )
            thought = structured.thought.strip()
            action = _normalize_action(structured.action)
            search_query = (structured.query or "").strip()
            
            logger.info(f"Thought: {thought}")
            logger.info(f"Action: {action}")
            if search_query:
                logger.info(f"Query: {search_query}")
                
        except Exception as e:
            logger.warning("Structured output failed, falling back to text parsing", exc_info=True)
            try:
                thought_response = await llm.generate_async(
                    messages=[{"role": "user", "content": thought_prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                parsed = _parse_react_response(thought_response)
                thought = parsed["thought"]
                action = parsed["action"]
                search_query = parsed["query"]
                
                logger.info(f"Thought: {thought}")
                logger.info(f"Action: {action}")
                if search_query:
                    logger.info(f"Query: {search_query}")
                    
            except Exception as inner_exc:
                logger.error(f"Error generating thought: {inner_exc}")
                thought = "Готовий відповісти з наявним контекстом"
                action = ACTION_ANSWER
                search_query = ""
        
        # Validate the ReAct step
        is_valid, error_msg = _validate_react_step(thought, action, search_query)
        if not is_valid:
            logger.warning(f"Invalid ReAct step: {error_msg}. Defaulting to answer.")
            observation = f"Невалідний крок: {error_msg}. Переходжу до відповіді."
            steps.append({"thought": thought, "action": ACTION_ANSWER, "observation": observation})
            break

        if action == ACTION_ANSWER:
            observation = "Готово до генерації відповіді"
            steps.append({"thought": thought, "action": action, "observation": observation})
            break

        if action == ACTION_SEARCH:
            # Check for duplicate query
            query_normalized = search_query.lower().strip()
            if query_normalized in searched_queries:
                observation = f"Запит '{search_query}' вже використовувався. Переходжу до відповіді."
                logger.warning(observation)
                steps.append({"thought": thought, "action": ACTION_ANSWER, "observation": observation})
                break
            
            searched_queries.add(query_normalized)
            
            # Generate embedding for the search query (NOT using original message embedding!)
            try:
                query_vector = await embedder.embed(search_query)
                logger.info(f"Generated embedding for query: '{search_query}'")
            except Exception as e:
                observation = f"Помилка генерації embedding: {e}"
                logger.error(observation, exc_info=True)
                steps.append({"thought": thought, "action": action, "observation": observation})
                continue
            
            # Search in Qdrant with the NEW query embedding
            try:
                search_results = await qdrant.search_similar(
                    query_vector=query_vector,
                    top_k=3,
                    only_relevant=True,
                )

                formatted_results = []
                for hit in search_results:
                    payload = hit.get("payload") or {}
                    formatted_results.append({
                        "content": payload.get("fact") or "",
                        "score": hit.get("score", 0.0),
                        "source_msg_uid": payload.get("messageid") or payload.get("record_id") or "unknown",
                        "timestamp": payload.get("timestamp"),
                    })

                # Update context with new results
                retrieved_context.extend(formatted_results)
                context_text = _build_context_text(retrieved_context)

                observation = format_search_results(formatted_results)
                logger.info(f"Found {len(formatted_results)} results for query '{search_query}'")
                
            except Exception as e:
                observation = f"Помилка пошуку: {e}"
                logger.error(observation, exc_info=True)

            steps.append({"thought": thought, "action": action, "observation": observation})
            continue

        # Safety fallback
        observation = "Приступаю до відповіді"
        steps.append({"thought": thought, "action": ACTION_ANSWER, "observation": observation})
        break

    logger.info(f"ReAct loop completed after {len(steps)} steps")
    logger.info(f"Total context items: {len(retrieved_context)}")

    return {
        "react_steps": steps,
        "retrieved_context": retrieved_context,  # Return updated context
    }
