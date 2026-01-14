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

    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="password123",
        description="Neo4j password"
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name"
    )

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

    # Embeddings Configuration
    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Sentence transformer model for embeddings (supports Ukrainian)"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embedding vectors"
    )
    use_hosted_embeddings: bool = Field(
        default=False,
        description="Use hosted Qwen embeddings instead of local sentence-transformers"
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