"""
Node 7: ReAct Loop
Implements ReAct (Reasoning + Acting) for complex task solving.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from pydantic import BaseModel

from agent.state import AgentState
from clients.llm_client import get_llm_client
from clients.graphiti_client import get_graphiti_client
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)

ACTION_ANSWER = "answer"
ACTION_SEARCH = "search"
ALLOWED_ACTIONS = {ACTION_ANSWER, ACTION_SEARCH}
ANSWER_KEYWORDS = ("готовий", "достатньо", "можу відповісти", "answer", "ready")
SEARCH_KEYWORDS = ("шукати", "знайти", "потрібно", "search", "find", "need")


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

Потрібен результат у форматі JSON з ключами:
  thought: короткий виклад логіки
  action: "answer" або "search"
  query: рядок запиту, тільки якщо action == "search"

Якщо контексту достатньо для відповіді, вибери action="answer".
Якщо контексту НЕМАЄ потрібної інформації, вибери action="search" і сформулюй запит.

JSON відповідь:"""

    return f"""🚫 TABULA RASA: Використовуй ТІЛЬКИ інформацію з контексту. НЕ використовуй pretrained knowledge.

Попередні кроки:
{history_text or "(немає)"}

Поточний контекст (що тебе навчили):
{context_text}

Завдання: {task}

Потрібен результат у форматі JSON з ключами:
  thought: короткий виклад логіки
  action: "answer" або "search"
  query: рядок запиту, тільки якщо action == "search"

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
        if action == ACTION_SEARCH and not query:
            query = thought
        return {"thought": thought, "action": action, "query": query}

    # Fallback: treat entire output as thought and infer action
    thought = response_text.strip()
    action = _infer_action_from_thought(thought)
    return {"thought": thought, "action": action, "query": ""}


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
    graphiti = await get_graphiti_client()

    # Build initial context
    retrieved_context = state.get("retrieved_context", [])
    context_text = _build_context_text(retrieved_context)

    task = state["message_text"]
    steps: List[Dict[str, Any]] = []

    for iteration in range(max_iterations):
        logger.info(f"\n--- ReAct Iteration {iteration + 1}/{max_iterations} ---")

        history_text = _build_history_text(steps) if steps else None
        thought_prompt = _build_thought_prompt(task, context_text, history_text, iteration)

        # Generate thought
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
            if action == ACTION_SEARCH and not search_query:
                search_query = thought
            logger.info(f"Thought: {thought}")
            logger.info(f"Action: {action}")
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
            except Exception as inner_exc:
                logger.error(f"Error generating thought: {inner_exc}")
                thought = "Готовий відповісти з наявним контекстом"
                action = ACTION_ANSWER
                search_query = ""

        if action == ACTION_ANSWER:
            observation = "Готово до генерації відповіді"
            steps.append({"thought": thought, "action": action, "observation": observation})
            break

        if action == ACTION_SEARCH:
            # Extract search query from thought if missing
            if not search_query:
                search_query = extract_search_query(thought) or task
            logger.info(f"Action: {action} - Query: {search_query}")

            try:
                search_results = await graphiti.search(query=search_query, limit=3)
                observation = format_search_results(search_results)
                logger.info(f"Observation: Found {len(search_results)} results")

                for result in search_results:
                    content = result.get("content", "") or result.get("fact", "")
                    if content:
                        context_text += f"\n{content}"
            except Exception as e:
                logger.error(f"Error during search: {e}")
                observation = f"Помилка пошуку: {e}"

            steps.append({"thought": thought, "action": action, "observation": observation})
            continue

        # Safety fallback
        observation = "Приступаю до відповіді"
        steps.append({"thought": thought, "action": ACTION_ANSWER, "observation": observation})
        break

    logger.info(f"ReAct loop completed after {len(steps)} steps")

    return {
        "react_steps": steps
    }
