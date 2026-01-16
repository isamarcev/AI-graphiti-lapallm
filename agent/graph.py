import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes.classify import orchestrator_node
from agent.nodes.check_conflicts import check_conflicts_node
from agent.nodes.resolve_conflicts import resolve_conflicts_node
from agent.nodes.store import store_knowledge_node
from agent.nodes.retrieve import retrieve_context_node
from agent.nodes.react import react_loop_node
from agent.nodes.generate_learn_response import generate_learn_response_node
from agent.nodes.generate_solve_response import generate_solve_response_node
from agent.nodes.validate import validate_response_node

logger = logging.getLogger(__name__)


def route_after_classify(state: AgentState) -> str:
    """
    Route after classify based on presence of memory_updates and intent.
    
    Flow:
    - Has memory_updates? -> process_memory (then check intent)
    - No memory_updates + solve intent? -> solve_direct (skip to retrieval)
    - No memory_updates + learn intent? -> learn_only (shouldn't happen, but handle)
    """
    memory_updates = state.get("memory_updates", [])
    intent = state.get("intent")
    
    logger.info(f"Routing after classify: memory_updates={len(memory_updates)}, intent={intent}")
    
    # If we have memory updates, process them first
    if memory_updates:
        logger.info("→ Routing to process_memory (knowledge_manager)")
        return "process_memory"
    
    # If pure solve (no memory updates), go directly to retrieval
    elif intent == "solve":
        logger.info("→ Routing to solve_direct (retrieve_context)")
        return "solve_direct"
    
    # Pure learn (no solve tasks) - shouldn't have empty memory_updates, but handle
    else:
        logger.info("→ Routing to learn_only (generate_learn_response)")
        return "learn_only"


def route_by_intent(state: AgentState) -> str:
    """
    Route after store_knowledge based on intent.
    
    CRITICAL: If intent is 'solve', we skip learn response generation
    because we'll generate solve response anyway (no token waste).
    
    Returns:
    - "learn" -> pure learn, generate learn response
    - "solve" -> mixed or pure solve, skip to retrieval for solve response ONLY
    """
    intent = state.get("intent", "solve")
    
    logger.info(f"Routing by intent after store_knowledge: {intent}")
    
    if intent == "learn":
        logger.info("→ Routing to generate_learn_response (pure learn)")
    else:
        logger.info("→ Routing to retrieve_context (solve - skip learn response to save tokens)")
    
    return intent


def create_agent_graph():
    """
    Create the improved agent graph with decomposer and conflict resolution.
    
    Flow:
    1. classify (orchestrator/decomposer)
    2a. If has memory_updates -> check_conflicts -> resolve_conflicts (if needed) -> store_knowledge
    2b. Then route by intent:
        - learn only -> generate_learn_response
        - solve -> retrieve_context -> react_loop -> generate_solve_response -> validate
    3. For pure solve (no memory updates) -> directly to retrieve_context
    """
    logger.info("Creating improved agent graph with conflict resolution chain...")

    workflow = StateGraph(AgentState)

    logger.debug("Adding nodes to graph...")
    # Core nodes
    workflow.add_node("classify", orchestrator_node)
    workflow.add_node("check_conflicts", check_conflicts_node)
    workflow.add_node("resolve_conflicts", resolve_conflicts_node)
    workflow.add_node("store_knowledge", store_knowledge_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("react_loop", react_loop_node)
    workflow.add_node("generate_solve_response", generate_solve_response_node)
    workflow.add_node("validate_response", validate_response_node)
    workflow.add_node("generate_learn_response", generate_learn_response_node)

    # Entry point
    workflow.set_entry_point("classify")
    logger.debug("Entry point set to 'classify'")

    # After classify: check if memory updates exist
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "process_memory": "check_conflicts",  # Start conflict resolution chain
            "solve_direct": "retrieve_context",
            "learn_only": "generate_learn_response"
        }
    )
    logger.debug("Added conditional routing from classify")

    # Conflict resolution chain
    workflow.add_conditional_edges(
        "check_conflicts",
        lambda state: "resolve_conflicts" if state.get("conflicts") else "store_knowledge",
        {
            "resolve_conflicts": "resolve_conflicts",
            "store_knowledge": "store_knowledge",
        },
    )
    workflow.add_edge("resolve_conflicts", "store_knowledge")
    logger.debug("Added conflict resolution chain")

    # After store_knowledge: route by intent
    workflow.add_conditional_edges(
        "store_knowledge",
        route_by_intent,
        {
            "learn": "generate_learn_response",
            "solve": "retrieve_context"
        }
    )
    logger.debug("Added conditional routing from store_knowledge")

    # Solve path (linear after retrieval)
    workflow.add_edge("retrieve_context", "react_loop")
    workflow.add_edge("react_loop", "generate_solve_response")
    workflow.add_edge("generate_solve_response", "validate_response")
    workflow.add_edge("validate_response", END)
    logger.debug("Added solve path edges")

    # Learn path
    workflow.add_edge("generate_learn_response", END)
    logger.debug("Added learn path edge")

    # Compile with checkpointing for conversation memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Improved agent graph compiled successfully with conflict resolution")

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
