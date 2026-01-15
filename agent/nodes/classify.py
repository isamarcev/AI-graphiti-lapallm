"""
Node 1 (Revised): Intent Decomposition & Orchestrator
Breaks down complex mixed messages into atomic actions:
1. Memory Updates (Learn/Correct)
2. Task Executions (Solve)
"""
import asyncio
import logging
from typing import List, Literal, Optional
import dspy
from pydantic import BaseModel, Field

from agent.state import AgentState
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)

# ============================================================================
# DSPy Models for Decomposition
# ============================================================================

class ActionItem(BaseModel):
    """Atomic unit of work derived from user message."""
    action_type: Literal["remember", "execute"] = Field(
        ...,
        description="'remember' for facts, rules, corrections. 'execute' for questions, tasks, requests."
    )
    content: str = Field(
        ...,
        description="The extracted text content relevant to this action."
    )
    context_needed: bool = Field(
        False,
        description="For 'execute': does this task require looking up information?"
    )

class DecomposeSignature(dspy.Signature):
    """
    Ти — аналітичний модуль універсального AI-агента.
    Твоя мета: Розбити вхідне повідомлення на список атомарних дій.

    Категорії дій:
    1. REMEMBER: Будь-яка інформація, яку користувач стверджує, надає як факт, або виправляє.
       - "Мій улюблений колір рожевий" -> remember
       - "Ти помилився, тут має бути Х" -> remember (це корекція знань)
       - "Ось документація: ..." -> remember

    2. EXECUTE: Будь-яке прохання щось зробити, порахувати, написати або відповісти.
       - "Напиши функцію..." -> execute
       - "Яка погода?" -> execute

    Важливо: Зберігай оригінальні формулювання та деталі в полі content.
    """
    message: str = dspy.InputField(desc="Raw user message containing mixed intents")
    actions: List[ActionItem] = dspy.OutputField(desc="List of atomic actions to take")

class DecomposerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(DecomposeSignature)

    def forward(self, message: str):
        return self.prog(message=message)

# ============================================================================
# Node Implementation
# ============================================================================


# Define a helper function to run inside the thread
def _run_decomposition(message: str, lm_instance: dspy.LM):
    """
    Runs the DSPy module inside a thread-safe context.
    This function blocks, so it must be run via asyncio.to_thread.
    """
    # Apply the settings ONLY for this specific execution scope
    with dspy.context(lm=lm_instance):
        decomposer = DecomposerModule()
        return decomposer(message)

@traceable(name="orchestrate_message")
async def orchestrator_node(state: AgentState) -> dict:
    """
    Node 1: Parses message and orchestrates memory updates BEFORE task execution.
    """
    logger.info("=== Orchestrator Node (Universal) ===")

    # 1. Initialize LM
    lm = dspy.LM(model=f"openai/{settings.model_name}", api_key=settings.api_key,
                 api_base=settings.lapa_url)

    # REMOVED: dspy.configure(lm=lm)

    message_text = state["message_text"]

    # 2. Decompose Message using Context Manager
    try:
        # We pass the lm object to the thread and apply context THERE
        prediction = await asyncio.to_thread(_run_decomposition, message_text, lm)

        actions = prediction.actions
        logger.info(f"Decomposed into {len(actions)} actions")
    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        # Fallback
        actions = [ActionItem(action_type="execute", content=message_text, context_needed=True)]

    logger.info(f"Decomposition result: {actions}")

    # 3. Separate actions into memory_updates and task_description
    memory_updates = []
    task_description = []

    for action in actions:
        if action.action_type == "remember":
            logger.info(f"Found REMEMBER action: {action.content}")
            memory_updates.append(action.content)

        elif action.action_type == "execute":
            logger.info(f"Found EXECUTE action: {action.content}")
            task_description.append(action.content)

    # 4. Determine intent and construct response
    final_task = "\n".join(task_description)

    if not final_task and memory_updates:
        logger.info("Pure LEARN scenario detected")
        return {
            "intent": "learn",
            "memory_updates": memory_updates,
            "original_message": message_text
        }

    logger.info(
        f"SOLVE scenario detected (memory_updates: {len(memory_updates)}, tasks: {len(task_description)})")
    return {
        "intent": "solve",
        "message_text": final_task,
        "original_message": message_text,
        "memory_updates": memory_updates
    }