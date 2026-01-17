"""
Node 7: ReAct Loop with Structured Output and Plugin-based Tools
Implements ReAct (Reasoning + Acting) with clean architecture.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from agent.helpers import format_search_results
from agent.state import AgentState
from clients.hosted_embedder import get_embedder, HostedQwenEmbedder
from clients.llm_client import get_llm_client
from clients.qdrant_client import QdrantClient
from config.settings import settings
from langsmith import traceable

logger = logging.getLogger(__name__)


# ============================================================================
# Structured Output Models
# ============================================================================

class ActionType(str, Enum):
    """Available actions for the agent."""
    ANSWER = "answer"
    SEARCH = "search"


class ReactThought(BaseModel):
    """Structured output for ReAct reasoning step."""

    thought: str = Field(
        ...,
        description="Agent's reasoning (1-2 sentences)",
        min_length=5,
        max_length=500
    )
    action: ActionType = Field(
        ...,
        description="Action to take: 'answer' if ready to respond, 'search' to find information"
    )
    tool_name: Optional[str] = Field(
        None,
        description="Name of the tool to use (required if action is not 'answer')"
    )
    tool_input: Optional[str] = Field(
        None,
        description="Input for the tool (e.g., search query, API parameters)",
        min_length=3,
        max_length=200
    )

    @field_validator("tool_input")
    @classmethod
    def validate_tool_input(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure tool_input is provided when action is not answer."""
        action = info.data.get("action")
        if action and action != ActionType.ANSWER:
            if not v or not v.strip():
                raise ValueError(f"tool_input is required when action is '{action}'")
        return v.strip() if v else None


class ReactStep(BaseModel):
    """Represents one step in the ReAct loop."""

    iteration: int
    thought: str
    action: ActionType
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    observation: str
    success: bool = True


# ============================================================================
# Tool System
# ============================================================================

class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool
    observation: str
    data: Optional[Any] = None


class Tool(ABC):
    """Base class for ReAct tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass

    @abstractmethod
    async def execute(self, tool_input: str, context: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given input."""
        pass


class SearchTool(Tool):
    """Search tool for finding information in vector database."""

    def __init__(self, qdrant_client: QdrantClient, embedder: HostedQwenEmbedder):
        self.qdrant = qdrant_client
        self.embedder = embedder
        self.searched_queries: Set[str] = set()

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for information in memory using semantic search. Input: 2-5 keywords."

    async def execute(self, tool_input: str, context: Dict[str, Any]) -> ToolResult:
        """Execute search in vector database."""
        query = tool_input.strip().lower()

        # Check for duplicate queries
        if query in self.searched_queries:
            return ToolResult(
                success=False,
                observation=f"Query '{tool_input}' already used. Try different keywords."
            )

        self.searched_queries.add(query)

        try:
            # Generate embedding for search query
            query_vector = await self.embedder.embed(tool_input)
            logger.info(f"Generated embedding for query: '{tool_input}'")

            # Search in Qdrant
            search_results = await self.qdrant.search_similar(
                query_vector=query_vector,
                top_k=3,
                only_relevant=True,
            )

            # Format results
            formatted_results = []
            for hit in search_results:
                payload = hit.get("payload") or {}
                formatted_results.append({
                    "content": payload.get("fact") or "",
                    "score": hit.get("score", 0.0),
                    "source_msg_uid": (
                        payload.get("messageid") or
                        payload.get("record_id") or
                        "unknown"
                    ),
                    "timestamp": payload.get("timestamp"),
                })

            observation = format_search_results(formatted_results)
            logger.info(f"Found {len(formatted_results)} results for query '{tool_input}'")

            return ToolResult(
                success=True,
                observation=observation,
                data=formatted_results
            )

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                observation=f"Search error: {str(e)}"
            )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools with descriptions."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def get_tools_description(self) -> str:
        """Get formatted description of all tools for LLM."""
        tools_list = self.list_tools()
        if not tools_list:
            return "No tools available."

        descriptions = []
        for tool in tools_list:
            descriptions.append(f"- {tool['name']}: {tool['description']}")

        return "\n".join(descriptions)


# ============================================================================
# Prompt Building
# ============================================================================

class PromptBuilder:
    """Builds prompts for ReAct agent."""

    @staticmethod
    def build_context_text(retrieved_context: List[Dict[str, Any]]) -> str:
        """Format retrieved context for prompt."""
        if not retrieved_context:
            return "(порожньо - нічого не навчили)"

        return "\n".join(
            f"(message_uid={ctx.get('source_msg_uid', 'unknown')}): {ctx.get('content', '')}"
            for i, ctx in enumerate(retrieved_context)
        )

    @staticmethod
    def build_history_text(steps: List[ReactStep]) -> str:
        """Format ReAct history for prompt."""
        if not steps:
            return "(немає попередніх кроків)"

        history_lines = []
        for step in steps:
            history_lines.append(
                f"Крок {step.iteration}:\n"
                f"  Думка: {step.thought}\n"
                f"  Дія: {step.action.value}"
            )
            if step.tool_name:
                history_lines.append(f"  Інструмент: {step.tool_name}")
            if step.tool_input:
                history_lines.append(f"  Вхід: {step.tool_input}")

            # Truncate long observations
            obs = step.observation
            if len(obs) > 200:
                obs = obs[:200] + "..."
            history_lines.append(f"  Результат: {obs}\n")

        return "\n".join(history_lines)

    @staticmethod
    def build_thought_prompt(
        task: str,
        context_text: str,
        history_text: str,
        tools_description: str,
        iteration: int
    ) -> str:
        """Build prompt for generating next ReAct step."""

        base_instructions = """🚫 TABULA RASA: У тебе НУЛЬОВІ знання про предметну область.
Використовуй ТІЛЬКИ інформацію з контексту нижче. НЕ використовуй pretrained knowledge."""

        tools_section = f"""
ДОСТУПНІ ІНСТРУМЕНТИ:
{tools_description}

Щоб використати інструмент, встанови action відмінним від "answer" і вкажи:
- tool_name: назва інструменту
- tool_input: конкретний вхід для інструменту
"""

        rules = """
ПРАВИЛА:
- Якщо в контексті Є достатня інформація → action="answer"
- Якщо в контексті НЕМАЄ потрібної інформації → обери відповідний інструмент
- tool_input має бути конкретним (2-5 ключових слів для пошуку), НЕ повним реченням
- НЕ повторюй попередні запити, шукай щось нове
- Якщо інструмент не дав результатів, спробуй інший підхід або відповідай з наявним контекстом
"""

        examples = """
ПРИКЛАДИ:
{"thought": "В контексті немає інформації про столицю", "action": "search", "tool_name": "search", "tool_input": "столиця України"}
{"thought": "Контекст містить відповідь про Київ", "action": "answer"}
{"thought": "Треба дізнатись про улюблену їжу", "action": "search", "tool_name": "search", "tool_input": "улюблена їжа користувача"}
"""

        if iteration == 0:
            # First iteration - no history
            return f"""{base_instructions}

Контекст з пам'яті (що тебе навчили):
{context_text}

Завдання: {task}
{tools_section}
{rules}
{examples}

Твоя відповідь (JSON з полями thought, action, tool_name?, tool_input?):"""

        # Subsequent iterations - include history
        return f"""{base_instructions}

Попередні кроки:
{history_text}

Поточний контекст (що тебе навчили):
{context_text}

Завдання: {task}
{tools_section}
{rules}

Твоя відповідь (JSON з полями thought, action, tool_name?, tool_input?):"""


# ============================================================================
# ReAct Agent
# ============================================================================

class ReactAgent:
    """ReAct agent with structured output and plugin-based tools."""

    def __init__(
        self,
        llm_client,
        tool_registry: ToolRegistry,
        max_iterations: int = 3
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.prompt_builder = PromptBuilder()

    async def generate_thought(
        self,
        prompt: str
    ) -> Optional[ReactThought]:
        """Generate next ReAct step using structured output."""
        try:
            # Try structured output first
            structured = await self.llm.generate_async(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.01,
                max_tokens=300,
                response_format=ReactThought
            )

            logger.info(f"Thought: {structured.thought}")
            logger.info(f"Action: {structured.action.value}")
            if structured.tool_name:
                logger.info(f"Tool: {structured.tool_name}")
            if structured.tool_input:
                logger.info(f"Input: {structured.tool_input}")

            return structured

        except Exception as e:
            logger.error(f"Failed to generate structured thought: {e}", exc_info=True)
            return None

    async def execute_action(
        self,
        thought: ReactThought,
        context: Dict[str, Any]
    ) -> ToolResult:
        """Execute the action specified in thought."""

        # If action is answer, we're done
        if thought.action == ActionType.ANSWER:
            return ToolResult(
                success=True,
                observation="Готово до генерації відповіді"
            )

        # Execute tool
        if not thought.tool_name:
            return ToolResult(
                success=False,
                observation="No tool specified for non-answer action"
            )

        tool = self.tools.get(thought.tool_name)
        if not tool:
            return ToolResult(
                success=False,
                observation=f"Unknown tool: {thought.tool_name}"
            )

        return await tool.execute(thought.tool_input or "", context)

    async def run(
        self,
        task: str,
        initial_context: List[Dict[str, Any]]
    ) -> tuple[List[ReactStep], List[Dict[str, Any]]]:
        """
        Run ReAct loop.

        Returns:
            (steps, updated_context) tuple
        """
        steps: List[ReactStep] = []
        retrieved_context = initial_context.copy()

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- ReAct Iteration {iteration + 1}/{self.max_iterations} ---")

            # Build prompt
            context_text = self.prompt_builder.build_context_text(retrieved_context)
            history_text = self.prompt_builder.build_history_text(steps)
            tools_desc = self.tools.get_tools_description()

            prompt = self.prompt_builder.build_thought_prompt(
                task=task,
                context_text=context_text,
                history_text=history_text,
                tools_description=tools_desc,
                iteration=iteration
            )

            # Generate thought
            thought = await self.generate_thought(prompt)
            if not thought:
                # Fallback to answer if generation fails
                steps.append(ReactStep(
                    iteration=iteration + 1,
                    thought="Не вдалося згенерувати думку, відповідаю з наявним контекстом",
                    action=ActionType.ANSWER,
                    observation="Готово до генерації відповіді",
                    success=False
                ))
                break

            # Execute action
            result = await self.execute_action(
                thought,
                context={"retrieved_context": retrieved_context}
            )

            # Create step record
            step = ReactStep(
                iteration=iteration + 1,
                thought=thought.thought,
                action=thought.action,
                tool_name=thought.tool_name,
                tool_input=thought.tool_input,
                observation=result.observation,
                success=result.success
            )
            steps.append(step)

            # Update context if tool returned data
            if result.data and isinstance(result.data, list):
                retrieved_context.extend(result.data)

            # Stop if action is answer or tool failed critically
            if thought.action == ActionType.ANSWER:
                break

            if not result.success and "already used" in result.observation:
                # Stop on duplicate query
                break

        logger.info(f"ReAct loop completed after {len(steps)} steps")
        logger.info(f"Total context items: {len(retrieved_context)}")

        return steps, retrieved_context


# ============================================================================
# Node Implementation
# ============================================================================

@traceable(name="react_loop")
async def react_loop_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 7: ReAct (Reasoning + Acting) loop with structured output.

    Iteratively:
    1. THOUGHT - what needs to be done (structured output)
    2. ACTION - use tool or generate answer
    3. OBSERVATION - result of action

    Args:
        state: Current agent state

    Returns:
        State update with react_steps and updated context
    """
    logger.info("=== ReAct Loop Node (Refactored) ===")

    # Initialize clients
    llm = get_llm_client()
    embedder = get_embedder()
    qdrant = QdrantClient()
    await qdrant.initialize()

    # Setup tool registry
    tool_registry = ToolRegistry()
    search_tool = SearchTool(qdrant, embedder)
    tool_registry.register(search_tool)

    # Future tools can be registered here:
    # tool_registry.register(CalculatorTool())
    # tool_registry.register(WeatherTool())
    # tool_registry.register(APITool())

    # Create agent
    max_iterations = getattr(settings, 'max_react_iterations', 3)
    agent = ReactAgent(llm, tool_registry, max_iterations)

    # Run ReAct loop
    retrieved_context = state.get("retrieved_context", [])
    task = state["message_text"]

    steps, updated_context = await agent.run(task, retrieved_context)

    # Convert steps to dict format for state
    steps_dict = [
        {
            "iteration": step.iteration,
            "thought": step.thought,
            "action": step.action.value,
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "observation": step.observation,
            "success": step.success
        }
        for step in steps
    ]

    return {
        "react_steps": steps_dict,
        "retrieved_context": updated_context,
    }