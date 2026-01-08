"""
State definition for LangGraph agent.
Defines the structure of data that flows through the graph.
"""

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from datetime import datetime


class AgentState(TypedDict):
    """
    State for the conversational agent with graph memory.

    This state is passed between nodes in the LangGraph and maintains
    the conversation context and retrieved memories.
    """

    # Conversation messages (accumulated over time)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Retrieved context from Graphiti memory
    retrieved_context: Optional[str]

    # User identification for memory isolation
    user_id: str

    # Session identifier for grouping conversations
    session_id: str

    # Timestamp of current interaction
    timestamp: Optional[datetime]

    # Intermediate data for processing
    current_query: Optional[str]

    # Flag to indicate if memory update is needed
    needs_memory_update: bool

    # Search results from Graphiti (for debugging/inspection)
    search_results: Optional[list]

    # Number of messages in this conversation
    message_count: int


def create_initial_state(
    user_id: str,
    session_id: str,
    initial_message: Optional[BaseMessage] = None
) -> AgentState:
    """
    Create an initial state for a new conversation.

    Args:
        user_id: Unique identifier for the user
        session_id: Session identifier
        initial_message: Optional first message to start with

    Returns:
        Initial AgentState
    """
    messages = [initial_message] if initial_message else []

    return AgentState(
        messages=messages,
        retrieved_context=None,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now(),
        current_query=None,
        needs_memory_update=False,
        search_results=None,
        message_count=len(messages)
    )


def get_last_user_message(state: AgentState) -> Optional[str]:
    """
    Extract the last user message from state.

    Args:
        state: Current agent state

    Returns:
        Content of the last user message, or None
    """
    for message in reversed(state["messages"]):
        if message.type == "human":
            return message.content
    return None


def get_conversation_history(state: AgentState, limit: Optional[int] = None) -> str:
    """
    Format conversation history as a string.

    Args:
        state: Current agent state
        limit: Optional limit on number of messages to include

    Returns:
        Formatted conversation history
    """
    messages = state["messages"]
    if limit:
        messages = messages[-limit:]

    history_lines = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        history_lines.append(f"{role}: {msg.content}")

    return "\n".join(history_lines)
