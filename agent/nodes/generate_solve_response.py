"""
Node: Generate Solve Response
Генерує відповідь для SOLVE режиму на основі retrieved context.
"""

import logging
from typing import Dict, Any, List
from agent.state import AgentState
from clients.llm_client import get_llm_client
from models.schemas import RetrievedContext, ReactStep, SolveResponse
from langsmith import traceable

logger = logging.getLogger(__name__)


def _build_context_section(retrieved_context: List[RetrievedContext]) -> tuple[str, dict[str, RetrievedContext]]:
    """
    Build formatted context section з типізованих RetrievedContext.

    Args:
        retrieved_context: Typed list of retrieved context

    Returns:
        tuple: (formatted_text, uid_to_context_map)
    """
    if not retrieved_context:
        return "", {}

    context_parts = []
    uid_map = {}

    for ctx in retrieved_context:
        # Типізовані поля - чіткі типи без .get()
        uid = ctx.source_msg_uid
        content = ctx.content
        score = ctx.score

        context_parts.append(
            f"[{uid}] (релевантність: {score:.2f}):\n. Content {content}"
        )
        uid_map[uid] = ctx

    formatted_text = "\n\n".join(context_parts)
    return formatted_text, uid_map


def _build_reasoning_summary(react_steps: List[ReactStep]) -> str:
    """
    Build reasoning summary з типізованих ReactStep.

    Args:
        react_steps: Typed list of ReAct steps

    Returns:
        Formatted reasoning summary
    """
    if not react_steps:
        return "Прямий пошук"

    reasoning_lines = []
    for i, step in enumerate(react_steps, 1):
        # Типізовані поля
        thought = step.thought
        action = step.action
        reasoning_lines.append(f"Крок {i}: {thought} [Дія: {action}]")

    return "\n".join(reasoning_lines)


def _build_prompt_with_context(message_text: str, context_text: str, reasoning_summary: str, available_uids: List[str]) -> str:
    """
    Build prompt для випадку коли Є контекст.

    Domain-agnostic: без згадок про код, структури, алгоритми.
    No template examples: LLM повинен генерувати власні відповіді.
    """
    uids_list = ", ".join(available_uids)

    return f"""Ти відповідаєш на запит користувача базуючись ВИКЛЮЧНО на наданій інформації.

**КРИТИЧНІ ПРАВИЛА:**
1. Використовуй ТІЛЬКИ інформацію з джерел нижче
2. У полі `referenced_sources` вкажи список UID джерел (msg-XXX), які ти використав
3. Якщо в джерелах недостатньо інформації - встанови `has_sufficient_info: false`
4. НЕ додавай інформацію, якої немає в джерелах
5. Якщо джерела суперечать - вкажи це у відповіді
6. Мати structured output з валідним закінченням генерації. Наприклад:
{{
  "response": "Маю недостатні вказівки."
}}

**ДОСТУПНІ ДЖЕРЕЛА:**
{context_text}

**ДОСТУПНІ UIDs:** {uids_list}

**ПРОЦЕС МІРКУВАННЯ:**
{reasoning_summary}

**ЗАПИТ КОРИСТУВАЧА:**
{message_text}



Відповідай українською мовою. Вкажи у `referenced_sources` всі використані UID."""


def _build_prompt_no_context(message_text: str) -> str:
    """
    Build prompt для випадку коли НЕМАЄ контексту.

    Чітко повідомляємо про відсутність інформації без template examples.
    """
    return f"""Ти відповідаєш на запит користувача.

**ВАЖЛИВО:** У твоїй базі знань НЕМАЄ інформації для відповіді на цей запит.

Твоя відповідь має:
1. Чітко сказати що ти не маєш необхідної інформації
2. Попросити користувача надати потрібну інформацію
3. Бути чесним про обмеження твоїх знань
4. Мати structured output з валідним закінченням генерації. Наприклад:
{{
  "response": "Маю недостатні вказівки."
}}

**ЗАПИТ КОРИСТУВАЧА:**
{message_text}

Відповідай українською мовою."""


@traceable(name="generate_solve_response")
async def generate_solve_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Генерує відповідь для SOLVE режиму на основі retrieved context.

    КРИТИЧНО: Використовує типізовані RetrievedContext і ReactStep.
    Кожне твердження має мати reference до джерела.

    Args:
        state: Current agent state з retrieved_context та react_steps

    Returns:
        State update with:
        - response: final answer з citations
        - references: list of source message UIDs
        - reasoning: ReAct reasoning summary
    """
    logger.info("=== Generate Solve Response Node ===")

    llm = get_llm_client()

    # ТИПІЗОВАНИЙ доступ до даних з runtime конвертацією dict → Pydantic
    retrieved_context_raw = state["retrieved_context"]
    retrieved_context: List[RetrievedContext] = [
        RetrievedContext(**ctx) if isinstance(ctx, dict) else ctx
        for ctx in retrieved_context_raw
    ]

    react_steps_raw = state["react_steps"]
    react_steps: List[ReactStep] = [
        ReactStep(**step) if isinstance(step, dict) else step
        for step in react_steps_raw
    ]

    message_text: str = state["message_text"]
    message_uid: str = state["message_uid"]

    # Build context section
    context_text, uid_map = _build_context_section(retrieved_context)

    # Build reasoning summary
    reasoning_summary = _build_reasoning_summary(react_steps)

    # Choose temperature: нижча для відповідей з контекстом (fact-based)
    temperature = 0.3 if context_text else 0.5

    # Build appropriate prompt з structured output
    if context_text:
        available_uids = list(uid_map.keys())
        prompt = _build_prompt_with_context(message_text, context_text, reasoning_summary, available_uids)
    else:
        prompt = _build_prompt_no_context(message_text)

    logger.debug(f"Context available: {bool(context_text)}, temperature: {temperature}")

    try:
        # Використовуємо STRUCTURED OUTPUT для гарантованого формату
        structured_response: SolveResponse = await llm.generate_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
            response_format=SolveResponse  # 🔥 Автоматична валідація формату
        )

        logger.debug(f"Generated response: {structured_response.response[:200]}...")
        logger.debug(f"Referenced sources from LLM: {structured_response.referenced_sources}")

        # Verify що всі використані UIDs є в retrieved context
        valid_uids = set()
        for uid in structured_response.referenced_sources:
            if uid in uid_map:
                valid_uids.add(uid)
                logger.debug(f"Valid source: {uid}")
            else:
                logger.warning(f"LLM referenced unknown source: {uid}")

        # Додати current message UID як reference
        references = list(valid_uids)
        if message_uid not in references:
            references.append(message_uid)

        logger.info(f"Generated solve response with {len(references)} references (has_sufficient_info: {structured_response.has_sufficient_info})")

        return {
            "response": structured_response.response,
            "references": references,
            "reasoning": reasoning_summary
        }

    except Exception as e:
        logger.error(f"Error generating solve response: {e}", exc_info=True)
        return {
            "response": f"Помилка при генерації відповіді: {str(e)}",
            "references": [message_uid],
            "reasoning": None
        }