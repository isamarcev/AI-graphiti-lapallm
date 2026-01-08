"""
LangGraph assembly - connecting nodes into a conversational agent flow.
Flow: START → retrieve_memory → generate_response → save_to_memory → END
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import retrieve_memory_node, generate_response_node, save_to_memory_node

logger = logging.getLogger(__name__)


def create_agent_graph():
    """
    Create and compile the LangGraph agent.

    Returns:
        Compiled graph ready for execution
    """
    logger.info("Creating agent graph...")

    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("save_to_memory", save_to_memory_node)

    # Define edges (flow)
    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "generate_response")
    workflow.add_edge("generate_response", "save_to_memory")
    workflow.add_edge("save_to_memory", END)

    # Compile with checkpointing for conversation memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Agent graph compiled successfully")

    return app


# Global agent instance
_agent_app = None


def get_agent_app():
    """Get or create the global agent application."""
    global _agent_app
    if _agent_app is None:
        _agent_app = create_agent_graph()
    return _agent_app
