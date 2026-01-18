"""
Configuration settings for the Graphiti + LangGraph + Lapa LLM demo.
Uses pydantic-settings for environment variable management.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    lapa_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible API"
    )
    api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM (usually not required for local servers)"
    )
    model_name: str = Field(
        default="lapa",
        description="Model name/path for Lapa LLM"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for LLM generation"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM responses"
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        description="Qdrant REST endpoint (e.g., http://qdrant:6333)"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (if auth enabled)"
    )
    qdrant_collection: str = Field(
        default="facts",
        description="Qdrant collection name for fact storage"
    )
    qdrant_distance: str = Field(
        default="cosine",
        description="Vector distance metric: cosine | dot | euclid"
    )

    # SQLite Database Configuration
    database_path: str = Field(
        default="./data/messages.db",
        description="Path to SQLite database file"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (SQL logging, etc.)"
    )

    @property
    def database_url(self) -> str:
        """Construct async SQLite URL for SQLAlchemy."""
        return f"sqlite+aiosqlite:///{self.database_path}"

    # Graphiti Configuration
    graphiti_temperature: float = Field(
        default=0.0,
        description="Temperature for Graphiti LLM (lower for more stable JSON generation)"
    )
    graphiti_max_episode_length: int = Field(
        default=10000,
        description="Maximum episode length for Graphiti"
    )
    graphiti_search_limit: int = Field(
        default=10,
        description="Number of results to retrieve from Graphiti search"
    )
    graphiti_relevance_threshold: float = Field(
        default=0.7,
        description="Minimum relevance score for retrieved memories"
    )
    graphiti_max_concurrent_llm: int = Field(
        default=10,
        description="Max concurrent LLM calls in Graphiti (for entity processing parallelization)"
    )
    graphiti_max_concurrent_embeddings: int = Field(
        default=20,
        description="Max concurrent embedding calls in Graphiti"
    )

    # Custom extraction instructions for Graphiti (helps with JSON generation)
    graphiti_custom_instructions: str = Field(
        default="""CRITICAL: You MUST return valid JSON.
- Always close all strings with "
- Always close all objects with }
- Always close all arrays with ]
- Double-check your JSON before responding.
- Keep responses concise.""",
        description="Custom instructions added to Graphiti extraction prompts for better JSON compliance"
    )

    # Embeddings Configuration
    embedding_model_name: str = Field(
        default="text-embedding-qwen",
        description="Embedding model name"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embedding vectors"
    )
    use_hosted_embeddings: bool = Field(
        default=False,
        description="Use hosted embeddings"
    )
    # LangSmith Tracing (optional)
    langchain_tracing_v2: bool = Field(
        default=False,
        alias="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing"
    )
    langchain_api_key: str = Field(
        default="",
        alias="LANGCHAIN_API_KEY",
        description="LangSmith API key"
    )
    langchain_project: str = Field(
        default="graphiti-lapa-demo",
        alias="LANGCHAIN_PROJECT",
        description="LangSmith project name"
    )

    # ReAct Configuration
    max_react_iterations: int = Field(
        default=3,
        description="Maximum iterations for ReAct loop"
    )

    # Reranker Configuration (Optional - для CPU-based reranking замість LLM)
    # Встановіть True щоб увімкнути, False щоб вимкнути
    enable_reranker: bool = Field(
        default=True,
        description="Enable cross-encoder reranker for context filtering (CPU-based, no LLM tokens)"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model for reranking. Рекомендовані: ms-marco-MiniLM-L-6-v2 (швидкий), ms-marco-MiniLM-L-12-v2 (точніший)"
    )
    reranker_top_k: int = Field(
        default=5,
        description="Кількість документів після reranking"
    )
    reranker_min_score: float = Field(
        default=0.1,
        description="Мінімальний score для включення документа (0-1 scale)"
    )

    # Phoenix Observability Configuration
    enable_phoenix: bool = Field(
        default=True,
        description="Enable Phoenix observability and tracing"
    )
    phoenix_collector_endpoint: str = Field(
        default="http://phoenix:6006",
        description="Phoenix collector endpoint for traces"
    )
    phoenix_project_name: str = Field(
        default="graphiti-lapa-agent",
        description="Project name for Phoenix traces"
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings