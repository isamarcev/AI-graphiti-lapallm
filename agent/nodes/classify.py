"""
Node 1: Intent Classification
Determines if message is LEARN (learning) or SOLVE (task).
Uses DSPy framework for declarative LLM programming.
"""

import logging
from typing import Dict, Any
import dspy

from agent.state import AgentState
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)


class ClassifyIntentSignature(dspy.Signature):
    """Класифікуй намір повідомлення: чи користувач навчає систему (LEARN), чи просить виконати завдання (SOLVE).

    Аналізуй тільки ФОРМУ повідомлення.
    """
    message: str = dspy.InputField(desc="Вхідне повідомлення від користувача")
    intent: str = dspy.OutputField(desc="Тільки 'learn' або 'solve'")
    reasoning: str = dspy.OutputField(desc="Коротке пояснення вибору")


class ClassifyIntent(dspy.Module):
    """DSPy module for intent classification."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyIntentSignature)

    def forward(self, message: str):
        """Classify message intent."""
        return self.classify(message=message)


# Global instances для кешування
_dspy_lm = None
_classify_module = None


def _get_dspy_lm():
    """Initialize DSPy language model for Lapa LLM via OpenAI-compatible API."""
    global _dspy_lm
    if _dspy_lm is None:
        # Використовуємо Lapa LLM через OpenAI-compatible API
        # LiteLLM потребує формат: openai/модель для OpenAI-compatible API
        _dspy_lm = dspy.LM(
            model=f"openai/{settings.model_name}",  # "openai/lapa" для LiteLLM
            api_key=settings.api_key or "EMPTY",
            api_base=settings.lapa_url,
            model_type="chat"
        )
        dspy.configure(lm=_dspy_lm, temperature=0.1)
        logger.info(f"DSPy LM configured: {settings.lapa_url} with model {settings.model_name}")
    return _dspy_lm


def _get_classify_module():
    """Get or create the classification module."""
    global _classify_module
    if _classify_module is None:
        _get_dspy_lm()  # Ensure LM is initialized
        _classify_module = ClassifyIntent()
        logger.info("DSPy ClassifyIntent module initialized")
    return _classify_module


@traceable(name="classify_intent")
async def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Classify intent of user message using DSPy framework.

    Determines whether the user is:
    - LEARN: Providing information/facts to learn
    - SOLVE: Asking to solve a task/question

    Args:
        state: Current agent state

    Returns:
        State update with intent
    """
    logger.info("=== Classify Intent Node (DSPy) ===")
    logger.info(f"Message: {state['message_text'][:100]}...")

    try:
        # Отримати DSPy модуль
        classifier = _get_classify_module()

        # Отримати повідомлення для класифікації
        message_text = state["message_text"]

        # Виконати класифікацію через DSPy
        # DSPy автоматично формує промпт з Signature (всі інструкції в desc полях)
        result = classifier(message_text)

        # Обробити результат від DSPy
        intent = result.intent.lower().strip()
        if intent not in ["learn", "solve"]:
            # Fallback: спробувати витягти з reasoning
            if "learn" in result.reasoning.lower():
                intent = "learn"
            elif "solve" in result.reasoning.lower():
                intent = "solve"
            else:
                intent = "solve"  # Default на SOLVE при неоднозначності

        # Парсинг confidence
        try:
            confidence = float(result.confidence)
            # Забезпечити що confidence в валідному діапазоні
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError, AttributeError):
            confidence = 0.7  # Default confidence

        reasoning = getattr(result, 'reasoning', '')

        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")
        logger.info(f"Reasoning: {reasoning}")

        return {
            "intent": intent,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Error classifying intent with DSPy: {e}", exc_info=True)
        # Default to SOLVE on error (safer to treat as task)
        return {
            "intent": "solve",
            "confidence": 0.5
        }
