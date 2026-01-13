"""
Graphiti client initialization with custom LLM provider (vLLM).
Handles graph memory storage and retrieval.

ВАЖНО: Для работы с vLLM/Ollama используется OpenAIGenericClient из graphiti_core.
Этот клиент поддерживает OpenAI-compatible endpoints с structured outputs.

Ссылки:
- https://help.getzep.com/graphiti/configuration/llm-configuration
- https://github.com/getzep/graphiti/blob/main/graphiti_core/llm_client/openai_client.py
"""

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from sentence_transformers import SentenceTransformer
from typing import Optional, List
import logging
import asyncio
from datetime import datetime
from clients.hosted_embedder import HostedQwenEmbedder

from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

# Try to import EmbedderClient
from graphiti_core.embedder import EmbedderClient

# Import reranker clients
from clients.noop_reranker import NoOpRerankerClient
from clients.hosted_reranker import HostedRerankerClient

from config.settings import settings

logger = logging.getLogger(__name__)


class CustomEmbedder(EmbedderClient):
    """
    Custom embedder using sentence-transformers for Ukrainian language support.
    """

    def __init__(self, model_name: str = None):
        """Initialize embedder with sentence-transformers model."""
        self.model_name = model_name or settings.embedding_model_name
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    async def create(self, input_data: str | List[str]) -> List[float]:
        """
        Create embedding for a SINGLE text (returns one vector).

        Args:
            input_data: Single string to embed (if list provided, uses first element)

        Returns:
            Single embedding vector as List[float]
        """
        # EmbedderClient.create() signature expects single vector output
        # If list provided, take first element
        if isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError("Empty list provided to create()")
            text = input_data[0]
        else:
            text = input_data

        # Run embedding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            self.model.encode,
            text
        )

        # Return single vector
        return embedding.tolist()

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for batch of texts.

        Args:
            input_data_list: List of strings to embed

        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self.model.encode,
            input_data_list
        )
        return embeddings.tolist()


class GraphitiClient:
    """
    Wrapper for Graphiti with custom LLM and embedder configuration.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
    ):
        """
        Initialize Graphiti client.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self.neo4j_user = neo4j_user or settings.neo4j_user
        self.neo4j_password = neo4j_password or settings.neo4j_password
        self.neo4j_database = neo4j_database or settings.neo4j_database

        self.graphiti: Optional[Graphiti] = None
        self._initialized = False

    async def initialize(self) -> "GraphitiClient":
        """
        Initialize Graphiti with custom LLM and embedder.

        Returns:
            Self for chaining
        """
        if self._initialized:
            logger.warning("Graphiti already initialized")
            return self

        logger.info("Initializing Graphiti client...")

        # Create LLMConfig for vLLM/Ollama
        # OpenAIGenericClient работает с OpenAI-compatible endpoints
        llm_config = LLMConfig(
            api_key=settings.vllm_api_key,  # vLLM не требует настоящий API ключ
            model=settings.vllm_model_name,
            small_model=settings.vllm_model_name,  # Используем ту же модель
            base_url=settings.vllm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.graphiti_max_episode_length
        )

        # Create LiteLLM-compatible client для Graphiti
        # Uses custom wrapper that fixes message roles for LiteLLM compatibility
        graphiti_llm = OpenAIGenericClient(config=llm_config)

        # Create embedder (hosted or local)
        if settings.use_hosted_embeddings:
            logger.info("Using hosted Qwen embeddings")
            embedder = HostedQwenEmbedder()
        else:
            logger.info("Using local sentence-transformers embeddings")
            embedder = CustomEmbedder()

        # Create reranker based on settings
        # IMPORTANT: Never pass None to Graphiti, as it will try to create
        # OpenAIRerankerClient which requires OPENAI_API_KEY

        # Legacy support: USE_RERANKER=true -> use BGE
        if settings.use_reranker and settings.reranker_type == "noop":
            reranker_type = "bge"
        else:
            reranker_type = settings.reranker_type.lower()

        if reranker_type == "bge":
            # BGE cross-encoder: accurate but VERY slow on CPU
            logger.info("🐢 Using BGE cross-encoder reranker (slow but accurate)")
            reranker = BGERerankerClient()
        elif reranker_type == "hosted":
            # Hosted reranker: uses Lapathon API (medium speed)
            logger.info("🌐 Using hosted API reranker (medium speed, API-based)")
            reranker = HostedRerankerClient(
                use_logprobs=settings.reranker_use_logprobs,
                max_concurrent=settings.reranker_max_concurrent
            )
        else:
            # NoOp reranker: returns passages in original order (fast)
            logger.info("🚀 Using NoOp reranker (maximum speed, no reranking)")
            reranker = NoOpRerankerClient()

        # Initialize Graphiti with parallelization config
        # Увага: database параметр не підтримується конструктором
        # Використовуємо лише uri, user, password
        self.graphiti = Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=graphiti_llm,
            embedder=embedder,
            cross_encoder=reranker,
            # Паралелізація LLM та embedding calls
            max_coroutines=settings.graphiti_max_concurrent_llm,
        )

        # Note: build_indices_and_constraints() is now called once
        # in app.py lifespan handler to avoid repeated builds per request
        self._initialized = True
        logger.info("Graphiti client initialized successfully")

        return self

    async def add_episode(
        self,
        episode_body: str,
        episode_name: str,
        source_description: str,
        reference_time: Optional[datetime] = None
    ) -> None:
        """
        Add an episode (conversation turn) to the graph memory.

        Args:
            episode_body: The content of the episode
            episode_name: Name/ID for this episode
            source_description: Description of the source (e.g., user_id)
            reference_time: Optional reference time (datetime object)
        """
        if not self._initialized or self.graphiti is None:
            raise RuntimeError("Graphiti not initialized. Call initialize() first.")

        try:
            await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=reference_time
            )
            logger.info(f"Episode added: {episode_name}")
        except Exception as e:
            logger.error(f"Failed to add episode: {e}", exc_info=True)
            raise

    async def add_episode_with_facts(
        self,
        episode_name: str,
        episode_body: str,
        source_description: str,
        reference_time: Optional[datetime] = None
    ):
        """
        Add episode with pre-extracted facts as JSON string.
        
        This method uses EpisodeType.json to pass pre-extracted facts as a
        JSON-serialized string, allowing Graphiti to process structured data
        without full text extraction, resulting in ~26s performance improvement.
        
        Args:
            episode_name: Episode identifier
            episode_body: JSON string with pre-extracted facts
            source_description: Source info (e.g., "user:123, uid:abc")
            reference_time: Temporal context (datetime object)
            
        Returns:
            Episode object with nodes and edges
            
        Reference:
            https://help.getzep.com/graphiti/core-concepts/adding-episodes
        """
        from graphiti_core.nodes import EpisodeType
        
        if not self._initialized or self.graphiti is None:
            raise RuntimeError("Graphiti not initialized. Call initialize() first.")
        
        try:
            # Pass JSON string with EpisodeType.json for optimized processing
            episode = await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,  # JSON string (not dict!)
                source=EpisodeType.json,
                source_description=source_description,
                reference_time=reference_time
            )
            logger.info(f"✓ Episode added with structured JSON: {episode_name}")
            return episode
        except Exception as e:
            logger.error(f"Failed to add episode with facts: {e}", exc_info=True)
            raise

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        relevance_threshold: Optional[float] = None
    ) -> List[dict]:
        """
        Search for relevant information in the graph memory.

        Args:
            query: Search query
            limit: Maximum number of results
            relevance_threshold: Minimum relevance score

        Returns:
            List of search results
        """
        if not self._initialized or self.graphiti is None:
            raise RuntimeError("Graphiti not initialized. Call initialize() first.")

        limit = limit or settings.graphiti_search_limit
        relevance_threshold = relevance_threshold or settings.graphiti_relevance_threshold

        try:
            results = await self.graphiti.search(
                query=query,
                num_results=limit
            )

            # Filter by relevance threshold if available
            filtered_results = []
            for result in results:
                # Graphiti results may have different structures depending on version
                # Adapt this based on actual result format
                if hasattr(result, 'score'):
                    if result.score >= relevance_threshold:
                        filtered_results.append({
                            'content': str(result),
                            'score': result.score
                        })
                else:
                    filtered_results.append({
                        'content': str(result),
                        'score': 1.0
                    })

            logger.info(f"Search returned {len(filtered_results)} results for query: {query}")
            return filtered_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_graph_stats(self) -> dict:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        if not self._initialized or self.graphiti is None:
            raise RuntimeError("Graphiti not initialized. Call initialize() first.")

        try:
            # Execute Cypher query to get stats
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            async with driver.session(database=self.neo4j_database) as session:
                # Count nodes
                node_result = await session.run("MATCH (n) RETURN count(n) as count")
                node_count = (await node_result.single())['count']

                # Count relationships
                rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = (await rel_result.single())['count']

            await driver.close()

            return {
                'node_count': node_count,
                'relationship_count': rel_count
            }

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {'node_count': 0, 'relationship_count': 0}

    async def close(self):
        """Close Graphiti connection."""
        if self.graphiti:
            await self.graphiti.close()
            logger.info("Graphiti connection closed")

    def __enter__(self):
        """Sync context manager entry."""
        raise RuntimeError("Use async context manager (async with) instead")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global instance
_graphiti_client: Optional[GraphitiClient] = None


async def get_graphiti_client() -> GraphitiClient:
    """
    Get or create the global Graphiti client instance.

    Returns:
        Initialized GraphitiClient
    """
    global _graphiti_client
    if _graphiti_client is None:
        _graphiti_client = GraphitiClient()
    await _graphiti_client.initialize()
    return _graphiti_client


def create_graphiti_client(**kwargs) -> GraphitiClient:
    """Create a new Graphiti client with custom parameters."""
    return GraphitiClient(**kwargs)
