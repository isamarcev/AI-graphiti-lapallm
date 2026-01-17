"""
Simplified ReAct Node using LangChain's built-in agent.
Much simpler and more reliable than custom implementation.
"""

import logging
from typing import Any, Dict, List

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from agent.state import AgentState
from clients.hosted_embedder import get_embedder
from clients.llm_client import get_llm_client
from clients.qdrant_client import QdrantClient
from langsmith import traceable

logger = logging.getLogger(__name__)


def create_search_tool(qdrant: QdrantClient, embedder):
    """Create a search tool for the ReAct agent."""
    
    @tool
    async def search(query: str) -> str:
        """Шукає інформацію в пам'яті за ключовими словами. Вхід: 2-5 ключових слів українською."""
        try:
            # Generate embedding
            query_vector = await embedder.embed(query)
            
            # Search in Qdrant
            results = await qdrant.search_similar(
                query_vector=query_vector,
                top_k=3,
                only_relevant=True,
            )
            
            if not results:
                return "Не знайдено релевантної інформації в пам'яті."
            
            # Format results
            formatted = []
            for i, hit in enumerate(results, 1):
                payload = hit.get("payload") or {}
                fact = payload.get("fact", "")
                description = payload.get("description", "")
                examples = payload.get("examples", [])
                source_id = payload.get("messageid") or payload.get("message_id") or "unknown"
                
                # Build result string
                result_parts = [f"{i}. Факт: {fact}"]
                if description:
                    result_parts.append(f"   Опис: {description}")
                if examples and isinstance(examples, list):
                    result_parts.append(f"   Приклади: {', '.join(examples)}")
                result_parts.append(f"   [джерело: {source_id}]")
                
                formatted.append("\n".join(result_parts))
            
            return "\n\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Помилка пошуку: {str(e)}"
    
    return search


@traceable(name="react_simple")
async def react_simple_node(state: AgentState) -> Dict[str, Any]:
    """
    Simplified ReAct node using LangChain's built-in agent.
    
    Args:
        state: Current agent state
        
    Returns:
        State update with response and references
    """
    logger.info("=== ReAct Simple Node ===")
    
    # Initialize clients
    llm_client = get_llm_client()
    llm = llm_client.get_langchain_llm()
    embedder = get_embedder()
    qdrant = QdrantClient()
    await qdrant.initialize()
    
    # Get initial context
    retrieved_context = state.get("retrieved_context", [])
    message_text = state["message_text"]
    
    # Format initial context
    context_text = ""
    if retrieved_context:
        context_parts = []
        for i, ctx in enumerate(retrieved_context, 1):
            content = ctx.get("content", "") or ctx.get("fact", "")
            description = ctx.get("description", "")
            examples = ctx.get("examples", [])
            source = ctx.get("source_msg_uid") or ctx.get("messageid") or "unknown"
            
            part = f"{i}. Факт: {content}"
            if description:
                part += f"\n   Опис: {description}"
            if examples and isinstance(examples, list):
                part += f"\n   Приклади: {', '.join(examples)}"
            part += f"\n   [джерело: {source}]"
            context_parts.append(part)
        context_text = "\n\n".join(context_parts)
    
    # Create tools
    search_tool = create_search_tool(qdrant, embedder)
    tools = [search_tool]
    
    # Create system prompt
    system_prompt = f"""Ти асистент, який відповідає на запитання користувача.

🚫 TABULA RASA: У тебе НУЛЬОВІ знання про предметну область.
Використовуй ТІЛЬКИ інформацію з контексту нижче або з пошуку. НЕ використовуй pretrained knowledge.

ПОЧАТКОВИЙ КОНТЕКСТ:
{context_text or "(порожньо)"}

ПРАВИЛА:
- Якщо в контексті Є інформація → відповідай одразу
- Якщо в контексті НЕМАЄ інформації → використай search tool
- Відповідай КОРОТКО (2-4 речення)
- Відповідай українською мовою
- ОБОВ'ЯЗКОВО вказуй джерела у форматі [джерело: X] для кожного факту, який використовуєш

ПРИКЛАД ВІДПОВІДІ:
"Щоб зробити страву солодкою, додай цукор [джерело: 2]. Для жовтого кольору використай куркуму [джерело: 3]."
"""
    
    # Create agent using langgraph prebuilt
    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=system_prompt
    )
    
    try:
        # Run agent with recursion limit (default is 25, we set higher for complex tasks)
        result = await agent_executor.ainvoke(
            {"messages": [("user", message_text)]},
            {"recursion_limit": 10}
        )
        
        # Extract messages from result
        messages = result.get("messages", [])
        
        # Build response from all messages
        response_parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                if content and isinstance(content, str):
                    response_parts.append(content)
        
        response = "\n\n".join(response_parts) if response_parts else "Не вдалося згенерувати відповідь"
        
        logger.info(f"ReAct completed with {len(messages)} messages")
        
        return {
            "response": response,
            "references": [],
            "reasoning": ""
        }
        
        
    except Exception as e:
        logger.error(f"ReAct agent error: {e}", exc_info=True)
        return {
            "response": f"Помилка: {str(e)}",
            "references": [],
            "reasoning": ""
        }
    finally:
        await qdrant.close()
