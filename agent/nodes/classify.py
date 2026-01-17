"""
Node 1: Simple Intent Classification
Determines if message is LEARN (store facts) or SOLVE (answer questions).
"""
import logging
from typing import Literal
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.storage.message_store import get_message_store
from clients.llm_client import get_llm_client
from langsmith import traceable

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Classification of user message intent."""
    intent: Literal["learn", "solve"] = Field(
        description="'learn' if user is providing facts/information to remember. 'solve' if user is asking questions or requesting tasks."
    )


@traceable(name="classify_intent")
async def orchestrator_node(state: AgentState) -> dict:
    """
    Classify message intent as LEARN or SOLVE.
    
    - LEARN: User provides facts, information, corrections to store
    - SOLVE: User asks questions or requests tasks
    """
    logger.info("=== Intent Classification Node ===")

    # Store raw message in KV store FIRST
    message_store = get_message_store()
    message_store.store(
        message_uid=state["message_uid"],
        raw_message=state["message_text"],
        timestamp=state["timestamp"]
    )
    logger.info(f"Stored raw message {state['message_uid']} in message store")

    message_text = state["message_text"]
    
    # Use LLM to classify intent
    llm_client = get_llm_client()
    
    try:
        response = await llm_client.generate_async(
            messages=[
                {
                    "role": "system",
                    "content": """Визнач намір повідомлення користувача:

запам'ятай: Користувач надає факти, інформацію,
Приклади:
- "Мій улюблений колір рожевий"
- "Запам'ятай, що Іван працює в Google"
- "Столиця Франції - Париж"

виріши: Користувач ставить запитання, просить виконати завдання або потребує щось зробити.
Приклади:
- "Який мій улюблений колір?"
- "Напиши функцію на мовы python для сортування списку. Умови: ..."
- "Яка сьогодні погода?"
- "Порахуй 2+2"
- "Виріши наступну задачу. Задача: ..."

Відповідай ТІЛЬКИ одним словом: 'запам'ятай' або 'виріши'."""
                },
                {
                    "role": "user",
                    "content": f"Повідомлення для класифікації:\n\n\"{message_text}\"\n\nЯкий це тип: 'запам'ятай' чи 'виріши'?"
                }
            ],
            temperature=0.001
        )
        
        # Parse intent from response
        intent_text = response.strip().lower()
        
        # Check for "learn" in English or Ukrainian variants
        if "learn" in intent_text or "запам'ятай" in intent_text or "навчатися" in intent_text or "навчання" in intent_text:
            intent = "learn"
        # Check for "solve" in English or Ukrainian variants
        elif "solve" in intent_text or "вирішити" in intent_text or "розв'язати" in intent_text or "вирішення" in intent_text or "виріши" in intent_text:
            intent = "solve"
        else:
            logger.warning(f"Unexpected intent response: {intent_text}, defaulting to 'learn'")
            intent = "learn"
            
            
        logger.info(f"Classified intent: {intent}")
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}, defaulting to 'solve'")
        intent = "solve"
    
    # Prepare response based on intent
    if intent == "learn":
        logger.info("LEARN intent: adding message to memory_updates")
        return {
            "intent": "learn",
            "memory_updates": [message_text]
        }
    else:
        logger.info("SOLVE intent: leaving memory_updates empty")
        return {
            "intent": "solve",
            "memory_updates": []
        }