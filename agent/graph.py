"""
LangGraph assembly - knowledge-centered agent з teach/solve branches.
Simplified architecture: classify -> (teach: store | solve: retrieve -> react -> generate)
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes.classify import classify_intent_node
from agent.nodes.store import store_knowledge_node
from agent.nodes.retrieve import retrieve_context_node
from agent.nodes.react import react_loop_node
from agent.nodes.generate_teach_response import generate_teach_response_node
from agent.nodes.generate_solve_response import generate_solve_response_node

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


def create_agent_graph():
    """
    Create knowledge-centered agent with separated teach/solve responses.

    Architecture:
    - Entry: classify intent (teach/solve)
    - TEACH branch: store_knowledge → generate_teach_response → END
    - SOLVE branch: retrieve_context → react_loop → generate_solve_response → END

    Each branch has dedicated response generation node.

    Returns:
        Compiled LangGraph application
    """
    logger.info("Creating knowledge-centered agent graph with separated responses...")

    workflow = StateGraph(AgentState)

    # Add nodes - окремі ноди для teach і solve responses
    logger.debug("Adding nodes to graph...")
    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("store_knowledge", store_knowledge_node)
    workflow.add_node("generate_teach_response", generate_teach_response_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
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
            "teach": "store_knowledge",
            "solve": "retrieve_context"
        }
    )
    logger.debug("Added conditional routing from classify")

    # TEACH path: store → generate confirmation → END
    workflow.add_edge("store_knowledge", "generate_teach_response")
    workflow.add_edge("generate_teach_response", END)
    logger.debug("TEACH path configured (store → teach_response)")

    # SOLVE path: retrieve → react → generate answer → END
    workflow.add_edge("retrieve_context", "react_loop")
    workflow.add_edge("react_loop", "generate_solve_response")
    workflow.add_edge("generate_solve_response", END)
    logger.debug("SOLVE path configured (retrieve → react → solve_response)")

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
