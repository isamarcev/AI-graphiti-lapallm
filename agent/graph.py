"""
LangGraph assembly - knowledge-centered agent з teach/solve branches.
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes.classify import classify_intent_node
from agent.nodes.extract import extract_facts_node
from agent.nodes.conflicts import check_conflicts_node
from agent.nodes.resolve import resolve_conflict_node
from agent.nodes.store import store_knowledge_node
from agent.nodes.retrieve import retrieve_context_node
from agent.nodes.react import react_loop_node
from agent.nodes.generate import generate_answer_node

logger = logging.getLogger(__name__)


def route_by_intent(state: AgentState) -> str:
    """
    Route після classify node.
    
    Returns:
        "teach" or "solve" based on classification
    """
    intent = state.get("intent")
    logger.info(f"Routing by intent: {intent}")
    
    if intent not in ["teach", "solve"]:
        logger.warning(f"Invalid intent '{intent}', defaulting to 'solve'")
        return "solve"
    
    return intent


def route_conflicts(state: AgentState) -> str:
    """
    Route після check_conflicts node.
    
    Returns:
        "resolve" if conflicts found, "store" otherwise
    """
    conflicts = state.get("conflicts", [])
    
    if conflicts:
        logger.info(f"Conflicts found: {len(conflicts)}, routing to resolve")
        return "resolve"
    
    logger.info("No conflicts, proceeding to store")
    return "store"


def create_agent_graph():
    """
    Create knowledge-centered agent з bidirectional flow.
    
    Architecture:
    - Entry: classify intent (teach/solve)
    - TEACH branch: extract facts → check conflicts → [resolve | store]
    - SOLVE branch: retrieve context → react loop → generate answer
    
    Implements key concepts from paper (Báez Santamaría, 2024):
    - Bidirectional knowledge flow (agent asks when conflicts detected)
    - Knowledge quality assessment (confidence, conflicts, completeness)
    - Epistemic awareness (reasoning about knowledge state)
    
    Returns:
        Compiled LangGraph application
    """
    logger.info("Creating knowledge-centered agent graph...")
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    logger.debug("Adding nodes to graph...")
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("extract_facts", extract_facts_node)
    workflow.add_node("check_conflicts", check_conflicts_node)
    workflow.add_node("resolve_conflict", resolve_conflict_node)
    workflow.add_node("store_knowledge", store_knowledge_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("react_loop", react_loop_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Entry point
    workflow.set_entry_point("classify")
    logger.debug("Entry point set to 'classify'")
    
    # Conditional routing після classify
    workflow.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "teach": "extract_facts",
            "solve": "retrieve_context"
        }
    )
    logger.debug("Added conditional routing from classify")
    
    # TEACH path
    workflow.add_edge("extract_facts", "check_conflicts")
    workflow.add_conditional_edges(
        "check_conflicts",
        route_conflicts,
        {
            "resolve": "resolve_conflict",
            "store": "store_knowledge"
        }
    )
    workflow.add_edge("resolve_conflict", END)  # Bidirectional - wait for user
    workflow.add_edge("store_knowledge", END)
    logger.debug("TEACH path configured")
    
    # SOLVE path
    workflow.add_edge("retrieve_context", "react_loop")
    workflow.add_edge("react_loop", "generate_answer")
    workflow.add_edge("generate_answer", END)
    logger.debug("SOLVE path configured")
    
    # Compile with checkpointing for conversation memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("Knowledge-centered agent graph compiled successfully")
    
    return app


# Global instance
_agent_app = None


def get_agent_app():
    """
    Get or create the global agent application.
    
    Returns:
        Compiled LangGraph agent
    """
    global _agent_app
    if _agent_app is None:
        logger.info("Initializing global agent application")
        _agent_app = create_agent_graph()
    return _agent_app
