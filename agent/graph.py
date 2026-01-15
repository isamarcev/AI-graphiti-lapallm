import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes.classify import classify_intent_node
from agent.nodes.store import store_knowledge_node
from agent.nodes.react import react_loop_node
from agent.nodes.generate_learn_response import generate_learn_response_node
from agent.nodes.generate_solve_response import generate_solve_response_node
from agent.nodes.check_conflicts import check_conflicts_node
from agent.nodes.resolve_conflicts import resolve_conflicts_node

logger = logging.getLogger(__name__)


def route_by_intent(state: AgentState) -> str:

    intent = state.get("intent")
    logger.info(f"Routing by intent: {intent}")
    
    if intent not in ["learn", "solve"]:
        logger.warning(f"Invalid intent '{intent}', defaulting to 'solve'")
        return "solve"
    
    return intent


def create_agent_graph():

    logger.info("Creating knowledge-centered agent graph with separated responses...")

    workflow = StateGraph(AgentState)

    logger.debug("Adding nodes to graph...")
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("check_conflicts", check_conflicts_node)
    workflow.add_node("resolve_conflicts", resolve_conflicts_node)
    workflow.add_node("store_knowledge", store_knowledge_node)
    workflow.add_node("generate_learn_response", generate_learn_response_node)
    workflow.add_node("react_loop", react_loop_node)
    workflow.add_node("generate_solve_response", generate_solve_response_node)

    # Entry point
    workflow.set_entry_point("classify")
    logger.debug("Entry point set to 'classify'")

    # Conditional routing від classify
    workflow.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "learn": "check_conflicts",
            "solve": "react_loop"
        }
    )
    logger.debug("Added conditional routing from classify")

    workflow.add_conditional_edges(
        "check_conflicts",
        lambda state: "resolve_conflicts" if state.get("conflicts") else "store_knowledge",
        {
            "resolve_conflicts": "resolve_conflicts",
            "store_knowledge": "store_knowledge",
        },
    )
    workflow.add_edge("resolve_conflicts", "store_knowledge")
    workflow.add_edge("store_knowledge", "generate_learn_response")
    workflow.add_edge("generate_learn_response", END)
    workflow.add_edge("react_loop", "generate_solve_response")
    workflow.add_edge("generate_solve_response", END)

    # Compile with checkpointing for conversation memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Knowledge-centered agent graph compiled successfully with separated responses")

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
