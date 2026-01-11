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

    # vLLM / LLM Configuration
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM server base URL with OpenAI-compatible API"
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM (usually not required for local servers)"
    )
    vllm_model_name: str = Field(
        default="lapa",
        description="Model name/path for Lapa LLM"
    )
    llm_temperature: float = Field(
        default=0.7,
        description="Temperature for LLM generation"
    )
    llm_max_tokens: int = Field(
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

    # Agent Configuration
    agent_system_prompt: str = Field(
        default="""Ти - корисний AI асистент, який розмовляє українською мовою.
Ти маєш доступ до своєї довготривалої пам'яті, яка зберігає факти про користувача
та попередні розмови. Використовуй цю інформацію для надання персоналізованих відповідей.

Завжди будь ввічливим, точним та корисним.""",
        description="System prompt for the agent"
    )
    max_conversation_history: int = Field(
        default=10,
        description="Number of previous messages to include in context"
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

    # Optional: Fallback OpenAI API (for testing without vLLM)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (fallback option)"
    )
    use_openai_fallback: bool = Field(
        default=False,
        description="Use OpenAI API instead of vLLM"
    )

    # ReAct Configuration
    max_react_iterations: int = Field(
        default=3,
        description="Maximum iterations for ReAct loop"
    )

    # Conflict detection
    conflict_detection_threshold: float = Field(
        default=0.7,
        description="Minimum confidence to detect conflict"
    )

    # Knowledge quality
    min_fact_confidence: float = Field(
        default=0.5,
        description="Minimum confidence for extracted facts"
    )

    # Debug settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug logging"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment/file."""
    global settings
    settings = Settings()
    return settings
