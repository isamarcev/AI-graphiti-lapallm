"""
Node 7: ReAct Loop
Implements ReAct (Reasoning + Acting) for complex task solving.
"""

import logging
from typing import Dict, Any, List

from agent.state import AgentState
from agent.helpers import extract_search_query, format_search_results
from clients.llm_client import get_llm_client
from clients.graphiti_client import get_graphiti_client
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)


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
    context_text = "\n".join([
        f"[{i}] ({ctx['source_msg_uid']}): {ctx['content']}"
        for i, ctx in enumerate(retrieved_context)
    ])

    task = state["message_text"]
    steps: List[Dict[str, Any]] = []

    for iteration in range(max_iterations):
        logger.info(f"\n--- ReAct Iteration {iteration + 1}/{max_iterations} ---")

        # Build prompt for thought
        if iteration == 0:
            thought_prompt = f"""Контекст з пам'яті:
{context_text if context_text else "(порожньо)"}

Завдання: {task}

Крок 1. Подумай: що потрібно зробити для вирішення завдання?
- Якщо контексту достатньо - скажи "готовий відповісти"
- Якщо потрібна додаткова інформація - опиши що саме треба знайти

Твоя думка:"""
        else:
            # Include previous steps in prompt
            history = "\n".join([
                f"Крок {i+1}:\n  Думка: {step['thought']}\n  Дія: {step['action']}\n  Результат: {step['observation'][:100]}..."
                for i, step in enumerate(steps)
            ])

            thought_prompt = f"""Попередні кроки:
{history}

Поточний контекст:
{context_text}

Завдання: {task}

Крок {iteration + 1}. Що далі?"""

        # Generate thought
        try:
            thought_response = await llm.generate_async(
                messages=[{"role": "user", "content": thought_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            thought = thought_response.strip()
            logger.info(f"Thought: {thought}")
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            thought = "Готовий відповісти з наявним контекстом"

        # Determine action
        thought_lower = thought.lower()

        if any(keyword in thought_lower for keyword in ["готовий", "достатньо", "можу відповісти", "answer", "ready"]):
            action = "answer"
            observation = "Готово до генерації відповіді"
            steps.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })
            logger.info(f"Action: {action}")
            break

        elif any(keyword in thought_lower for keyword in ["шукати", "знайти", "потрібно", "search", "find", "need"]):
            action = "search"

            # Extract search query from thought
            search_query = extract_search_query(thought)
            logger.info(f"Action: {action} - Query: {search_query}")

            # Search for additional context
            try:
                search_results = await graphiti.search(
                    query=search_query,
                    limit=3
                )

                observation = format_search_results(search_results)
                logger.info(f"Observation: Found {len(search_results)} results")

                # Add new results to context
                for result in search_results:
                    content = result.get('content', '') or result.get('fact', '')
                    context_text += f"\n{content}"

            except Exception as e:
                logger.error(f"Error during search: {e}")
                observation = f"Помилка пошуку: {e}"

            steps.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

        else:
            # Default: assume ready to answer
            action = "answer"
            observation = "Приступаю до відповіді"
            steps.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })
            logger.info(f"Action: {action} (default)")
            break

    logger.info(f"ReAct loop completed after {len(steps)} steps")

    return {
        "react_steps": steps
    }
