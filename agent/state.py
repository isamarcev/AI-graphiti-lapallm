"""
State definition for knowledge-centered agent.
Defines the structure of data that flows through the graph.
"""

from typing import TypedDict, Literal, Optional, List, Tuple, Dict, Any
from datetime import datetime
from models.schemas import RetrievedContext, ReactStep


class AgentState(TypedDict):

    # Input
    message_uid: str
    system_message_id: int
    message_text: str
    timestamp: datetime

    # Classification
    intent: Optional[Literal["learn", "solve"]]
    
    # Decomposition - for mixed messages
    memory_updates: List[str]  # "remember" actions from decomposer
    original_message: Optional[str]  # preserve original mixed message

    indexed_facts: List[Dict[str, Any]]  # indexed facts from decomposer
    # SOLVE path - TYPED structures
    retrieved_context: List[RetrievedContext]
    react_steps: List[ReactStep]
    conflicts: List[Tuple[str, str]]  # (message_id, fact)
    message_embedding: List[float]
    validation_attempts: int  # track validator retries

    # Output
    response: str
    references: List[str]  # message UIDs
    reasoning: Optional[str]


def create_initial_state(
    message_uid: str,
    message_text: str,
    system_message_id: int,
    user_id: str = "default_user",
    timestamp: Optional[datetime] = None
) -> AgentState:
    """
    Create an initial state for a new message.

    Args:
        message_uid: Unique identifier for the message
        system_message_id: Unique identifier for the message in the system
        message_text: Content of the message
        user_id: User identifier
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Initial AgentState with empty values
    """
    if timestamp is None:
        timestamp = datetime.now()

    return AgentState(
        message_uid=message_uid,
        system_message_id=system_message_id,
        message_text=message_text,
        user_id=user_id,
        timestamp=timestamp,
        intent=None,
        memory_updates=[],
        original_message=None,
        indexed_facts=[],
        retrieved_context=[],
        react_steps=[],
        conflicts=[],
        message_embedding=[],
        validation_attempts=0,
        response="",
        references=[],
        reasoning=None
    )
