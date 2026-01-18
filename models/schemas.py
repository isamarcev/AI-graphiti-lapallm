"""
Pydantic models for structured outputs from LLM.
These schemas ensure consistent data structures for agent operations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class AgentResponse(BaseModel):
    """Structured response from the agent."""

    response_text: str = Field(
        description="The main response text to display to the user"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level of the response (0.0 to 1.0)"
    )
    requires_memory_update: bool = Field(
        default=True,
        description="Whether this response contains information that should be stored in memory"
    )
    language: str = Field(
        default="uk",
        description="Response language code (uk for Ukrainian)"
    )


class ExtractedEntity(BaseModel):
    """Entity extracted from conversation for knowledge graph."""

    name: str = Field(
        description="Name of the entity"
    )
    entity_type: str = Field(
        description="Type of entity (person, place, organization, concept, etc.)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the entity"
    )


class ExtractedRelation(BaseModel):
    """Relationship between entities extracted from conversation."""

    source_entity: str = Field(
        description="Name of the source entity"
    )
    relation_type: str = Field(
        description="Type of relationship (likes, lives_in, works_at, knows, etc.)"
    )
    target_entity: str = Field(
        description="Name of the target entity"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship"
    )


class ConversationInsights(BaseModel):
    """Insights extracted from conversation for memory storage."""

    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="List of entities mentioned in the conversation"
    )
    relations: List[ExtractedRelation] = Field(
        default_factory=list,
        description="List of relationships between entities"
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description="Important facts to remember from this conversation"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Main topics discussed"
    )
    sentiment: Optional[str] = Field(
        default=None,
        description="Overall sentiment (positive, negative, neutral)"
    )


class MemorySearchQuery(BaseModel):
    """Query for searching in Graphiti memory."""

    query_text: str = Field(
        description="The search query text"
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )
    relevance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score"
    )


class MemorySearchResult(BaseModel):
    """Result from Graphiti memory search."""

    content: str = Field(
        description="Retrieved content from memory"
    )
    relevance_score: float = Field(
        description="Relevance score of this result"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When this memory was created"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source of this memory (episode name, user_id, etc.)"
    )


class ConversationMetadata(BaseModel):
    """Metadata about a conversation turn."""

    user_id: str = Field(
        description="Unique identifier for the user"
    )
    session_id: str = Field(
        description="Session identifier for grouping conversations"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of this conversation turn"
    )
    message_count: int = Field(
        default=0,
        description="Number of messages in this session"
    )


# Example schemas for specific use cases

class UserPreferences(BaseModel):
    """User preferences extracted from conversations."""

    preferences: dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of preference key-value pairs"
    )
    interests: List[str] = Field(
        default_factory=list,
        description="List of user interests"
    )
    language_preference: str = Field(
        default="uk",
        description="Preferred language for communication"
    )


class KnowledgeGraphNode(BaseModel):
    """Representation of a node in the knowledge graph."""

    id: str = Field(
        description="Unique identifier for the node"
    )
    name: str = Field(
        description="Display name of the node"
    )
    node_type: str = Field(
        description="Type of node (entity type)"
    )
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties of the node"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )


class KnowledgeGraphEdge(BaseModel):
    """Representation of an edge in the knowledge graph."""

    source_id: str = Field(
        description="ID of source node"
    )
    target_id: str = Field(
        description="ID of target node"
    )
    relation_type: str = Field(
        description="Type of relationship"
    )
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties of the edge"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )


class RetrievedContext(BaseModel):
    """
    Typed structure for retrieved context from Graphiti.
    Used in SOLVE path after retrieval.
    """

    content: str = Field(
        description="Retrieved text content from memory"
    )
    source_msg_uid: str = Field(
        description="UID of the message that is the source of this content"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When this content was originally stored"
    )
    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance score from retrieval (0.0 to 1.0)"
    )


class ReactStep(BaseModel):
    """
    Typed structure for a single ReAct reasoning step.
    Used in SOLVE path during react_loop.
    """

    thought: str = Field(
        description="Agent's thought/reasoning at this step"
    )
    action: str = Field(
        description="Action decided (e.g. 'search', 'answer', 'refine')"
    )
    observation: Optional[str] = Field(
        default=None,
        description="Result/observation from executing the action"
    )


class SolveResponse(BaseModel):
    """
    Structured output schema для SOLVE response.
    Використовується для гарантії формату відповіді від LLM.
    """

    response: str = Field(
        description="Фінальна відповідь на запит користувача українською мовою"
    )
    referenced_sources: List[str] = Field(
        description="Список UID джерел у форматі msg-XXX, які були використані у відповіді",
        default_factory=list
    )
    has_sufficient_info: bool = Field(
        description="True якщо в джерелах була достатня інформація, False якщо ні"
    )


class ContextRelevanceItem(BaseModel):
    """Оценка релевантности одного элемента контекста."""

    index: int = Field(
        description="Індекс елемента контексту (0-based)"
    )
    is_relevant: bool = Field(
        description="Чи релевантний цей контекст для відповіді на запит"
    )


class ContextActualizationResult(BaseModel):
    """Результат AI-фільтрації контексту."""

    items: List[ContextRelevanceItem] = Field(
        description="Оцінки релевантності для кожного елемента"
    )
    total_relevant: int = Field(
        description="Кількість релевантних елементів"
    )


class QueryAnalysis(BaseModel):
    """Аналіз запиту користувача для формування стратегії пошуку."""

    query_type: str = Field(
        description="Тип запиту: factual_question, task, explanation, comparison, other"
    )
    domain: str = Field(
        description="Предметна область запиту (їжа, робота, хобі, місце, загальне та ін.)"
    )
    key_entities: List[str] = Field(
        description="Ключові сутності у запиті (імена людей, назви місць, об'єкти)",
        default_factory=list
    )
    required_tools_or_methods: List[str] = Field(
        description="Інструменти, методи чи технології згадані в запиті (якщо є). Приклади: назви інструментів, методів, технологій, матеріалів",
        default_factory=list,
        max_items=3
    )
    information_needs: List[str] = Field(
        description="Чек-лист інформації, необхідної для повної відповіді",
        min_items=1,
        max_items=5
    )
    search_queries: List[str] = Field(
        description="Оптимізовані пошукові запити для збору потрібного контексту",
        min_items=1,
        max_items=5
    )
