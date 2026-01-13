"""
State definition for knowledge-centered agent.
Defines the structure of data that flows through the graph.
"""

from typing import TypedDict, Literal, Optional, List
from datetime import datetime
from models.schemas import RetrievedContext, ReactStep


class AgentState(TypedDict):
    """
    State для knowledge-centered agent з teach/solve paths.

    Simplified architecture:
    - TEACH path: user provides knowledge, agent stores directly to Graphiti
    - SOLVE path: agent retrieves and uses knowledge to solve tasks
    """

    # Input
    message_uid: str
    message_text: str
    user_id: str
    timestamp: datetime

    # Classification
    intent: Optional[Literal["teach", "solve"]]
    confidence: float

    # SOLVE path - TYPED structures
    retrieved_context: List[RetrievedContext]
    react_steps: List[ReactStep]

    # Output
    response: str
    references: List[str]  # message UIDs
    reasoning: Optional[str]


def create_initial_state(
    message_uid: str,
    message_text: str,
    user_id: str = "default_user",
    timestamp: Optional[datetime] = None
) -> AgentState:
    """
    Create an initial state for a new message.

    Args:
        message_uid: Unique identifier for the message
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
        message_text=message_text,
        user_id=user_id,
        timestamp=timestamp,
        intent=None,
        confidence=0.0,
        retrieved_context=[],
        react_steps=[],
        response="",
        references=[],
        reasoning=None
    )
