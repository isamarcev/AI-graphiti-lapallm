"""
State definition for knowledge-centered agent.
Defines the structure of data that flows through the graph.
"""

from typing import TypedDict, Literal, Optional, List
from datetime import datetime


class AgentState(TypedDict):
    """
    State для knowledge-centered agent з teach/solve paths.
    
    This state supports bidirectional knowledge flow:
    - TEACH path: user provides knowledge, agent learns
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
    
    # TEACH path
    extracted_facts: List[dict]  # [{subject, relation, object, confidence}]
    conflicts: List[dict]  # [{old_msg_uid, new_msg_uid, old_content, new_content, description, score}]
    conflict_resolved: bool
    confirmation_text: Optional[str]  # Generated confirmation message for teaching
    
    # SOLVE path
    retrieved_context: List[dict]  # [{content, source_msg_uid, timestamp, score}]
    react_steps: List[dict]  # [{thought, action, observation}]
    
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
        extracted_facts=[],
        conflicts=[],
        conflict_resolved=True,
        confirmation_text=None,
        retrieved_context=[],
        react_steps=[],
        response="",
        references=[],
        reasoning=None
    )
