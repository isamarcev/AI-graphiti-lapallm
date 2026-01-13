"""
LangGraph nodes for the conversational agent with graph memory.
Defines the processing steps: retrieve → generate → save.
"""

import logging
import time
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any

from agent.state import AgentState, get_last_user_message
from clients.llm_client import get_llm_client
from clients.graphiti_client import get_graphiti_client
from config.settings import settings

# LangSmith tracing
from langsmith import traceable

logger = logging.getLogger(__name__)


@traceable(name="retrieve_memory")
async def retrieve_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Retrieve relevant context from Graphiti memory.

    This node searches the graph memory for information relevant to
    the current user query.

    Args:
        state: Current agent state

    Returns:
        State updates with retrieved context
    """
    logger.info("=== Retrieve Memory Node ===")

    # Get the last user message
    user_message = get_last_user_message(state)
    if not user_message:
        logger.warning("No user message found")
        return {
            "retrieved_context": "",
            "search_results": [],
            "current_query": ""
        }

    logger.info(f"User query: {user_message[:100]}...")

    # === PROFILING START ===
    start_total = time.time()

    try:
        # Get Graphiti client
        start_client = time.time()
        graphiti = await get_graphiti_client()
        time_client = time.time() - start_client
        logger.info(f"⏱️  Graphiti client init: {time_client:.3f}s")

        # Search for relevant memories
        start_search = time.time()
        search_results = await graphiti.search(
            query=user_message,
            limit=settings.graphiti_search_limit,
            relevance_threshold=settings.graphiti_relevance_threshold
        )
        time_search = time.time() - start_search

        # Log search time with reranker status
        reranker_status = "WITH reranking" if settings.use_reranker else "WITHOUT reranking"
        logger.info(f"⏱️  Search ({reranker_status}): {time_search:.3f}s")

        # Format retrieved context
        if search_results:
            context_parts = []
            for i, result in enumerate(search_results, 1):
                content = result.get('content', '')
                score = result.get('score', 0.0)
                context_parts.append(f"{i}. [Score: {score:.2f}] {content}")

            retrieved_context = "\n".join(context_parts)
            logger.info(f"Retrieved {len(search_results)} memory results")
        else:
            retrieved_context = "Немає релевантних спогадів з попередніх розмов."
            logger.info("No relevant memories found")

        # === PROFILING END ===
        total_time = time.time() - start_total
        logger.info(f"⏱️  TOTAL retrieval time: {total_time:.3f}s")

        return {
            "retrieved_context": retrieved_context,
            "search_results": search_results,
            "current_query": user_message
        }

    except Exception as e:
        # Log error with timing
        total_time = time.time() - start_total
        logger.error(f"Error retrieving memory: {e} (took {total_time:.3f}s)")
        return {
            "retrieved_context": "Помилка при отриманні контексту.",
            "search_results": [],
            "current_query": user_message
        }


@traceable(name="generate_response")
async def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Generate response using LLM with retrieved context.

    This node uses the LLM to generate a response based on:
    - Current user message
    - Retrieved context from memory
    - Conversation history

    Args:
        state: Current agent state

    Returns:
        State updates with AI response
    """
    logger.info("=== Generate Response Node ===")

    # Get components
    user_message = get_last_user_message(state)
    retrieved_context = state.get("retrieved_context", "")
    llm_client = get_llm_client()

    # Build prompt with context
    system_prompt = settings.agent_system_prompt

    if retrieved_context and retrieved_context != "Немає релевантних спогадів з попередніх розмов.":
        system_prompt += f"\n\n=== Контекст з пам'яті ===\n{retrieved_context}\n=== Кінець контексту ==="

    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Add conversation history (last N messages)
    history_limit = min(settings.max_conversation_history, len(state["messages"]))
    for msg in state["messages"][-history_limit:]:
        role = "user" if msg.type == "human" else "assistant"
        messages.append({"role": role, "content": msg.content})

    logger.info(f"Generating response with {len(messages)} messages in context")

    try:
        # Generate response
        response_text = await llm_client.generate_async(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

        logger.info(f"Generated response: {response_text[:100]}...")

        # Create AI message
        ai_message = AIMessage(content=response_text)

        return {
            "messages": [ai_message],
            "needs_memory_update": True,
            "timestamp": datetime.now(),
            "message_count": state.get("message_count", 0) + 1
        }

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        error_message = AIMessage(
            content=f"Вибачте, сталася помилка при генерації відповіді: {str(e)}"
        )
        return {
            "messages": [error_message],
            "needs_memory_update": False
        }


@traceable(name="save_to_memory")
async def save_to_memory_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Save conversation episode to Graphiti memory.

    This node saves the current conversation turn (user message + AI response)
    to the graph memory for future retrieval.

    Args:
        state: Current agent state

    Returns:
        State updates confirming memory save
    """
    logger.info("=== Save to Memory Node ===")

    # Check if memory update is needed
    if not state.get("needs_memory_update", False):
        logger.info("Memory update not needed, skipping")
        return {"needs_memory_update": False}

    try:
        # Get last two messages (user + assistant)
        messages = state["messages"]
        if len(messages) < 2:
            logger.warning("Not enough messages to save")
            return {"needs_memory_update": False}

        # Get user and assistant messages
        user_msg = None
        ai_msg = None

        for msg in reversed(messages):
            if msg.type == "ai" and ai_msg is None:
                ai_msg = msg
            elif msg.type == "human" and user_msg is None:
                user_msg = msg

            if user_msg and ai_msg:
                break

        if not user_msg or not ai_msg:
            logger.warning("Could not find user/AI message pair")
            return {"needs_memory_update": False}

        # Format episode
        episode_body = f"""User: {user_msg.content}Assistant: {ai_msg.content}
"""

        # Create episode name from timestamp and user_id
        timestamp = state.get("timestamp", datetime.now())
        episode_name = f"{state['user_id']}_{timestamp.isoformat()}"

        # Get Graphiti client
        graphiti = await get_graphiti_client()
        # Save episode
        await graphiti.add_episode(
            episode_body=episode_body,
            episode_name=episode_name,
            source_description=f"user:{state['user_id']}, session:{state['session_id']}",
            reference_time=timestamp  # Pass datetime object directly
        )

        logger.info(f"Episode saved: {episode_name}")

        return {"needs_memory_update": False}

    except Exception as e:
        logger.error(f"Error saving to memory: {e}", exc_info=True)
        return {"needs_memory_update": False}
