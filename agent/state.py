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
    system_message_id: int | None
    message_text: str
    timestamp: datetime

    # Classification
    intent: Optional[Literal["learn", "solve"]]
    
    # Decomposition - for mixed messages
    memory_updates: List[str]  # "remember" actions from decomposer
    subtasks: List[str]  # decomposed subtasks from task decomposition

    indexed_facts: List[Dict[str, Any]]  # indexed facts from decomposer
    # SOLVE path - TYPED structures
    query_analysis: Optional[Dict[str, Any]]  # query analysis result
    retrieved_context: List[RetrievedContext]
    relevant_context: List[Dict[str, Any]]  # filtered context after actualization
    react_steps: List[ReactStep]
    conflicts: List[Tuple[str, str]]  # (message_id, fact)
    validation_attempts: int  # track validator retries

    # Output
    learn_response: str
    solve_response: str
    response: str
    references: List[str]  # message UIDs
    reasoning: Optional[str]

    plan: str
    search_queries: List[str]
    sources: List[str]


def create_initial_state(
    message_uid: str,
    message_text: str,
    system_message_id: int | None = None,
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
        system_message_id=system_message_id,
        message_text=message_text,
        user_id=user_id,
        timestamp=timestamp,
        intent=None,
        memory_updates=[],
        subtasks=[],
        indexed_facts=[],
        query_analysis=None,
        retrieved_context=[],
        relevant_context=[],
        react_steps=[],
        conflicts=[],
        validation_attempts=0,
        learn_response="",
        solve_response="",
        response="",
        references=[],
        reasoning=None,
        plan="",
        search_queries=[],
        sources=[]
    )
