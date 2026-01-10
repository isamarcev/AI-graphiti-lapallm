"""
Alternative embedder using hosted Qwen embeddings API.
Use this instead of CustomEmbedder when connecting to Lapathon hosted API.
"""

from typing import List
import asyncio
import logging
from openai import AsyncOpenAI

from graphiti_core.embedder import EmbedderClient
from config.settings import settings

logger = logging.getLogger(__name__)


class HostedQwenEmbedder(EmbedderClient):
    """
    Embedder using hosted text-embedding-qwen model.

    This embedder calls the Lapathon hosted API instead of using
    local sentence-transformers.
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model_name: str = None
    ):
        """
        Initialize hosted embedder.

        Args:
            base_url: API base URL (default from settings)
            api_key: API key (default from settings)
            model_name: Model name (default from settings.embedding_model_name)
        """
        # Hosted API doesn't use /v1 suffix, so strip it if present
        self.base_url = base_url or settings.vllm_base_url.rstrip('/v1')
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.embedding_model_name

        logger.info(f"Initializing hosted embedder: {self.model_name}")
        logger.info(f"API URL: {self.base_url}")

        # Create OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        # Qwen embeddings dimension (check with test_hosted_api.py)
        # Common dimensions: 768, 1024, 1536
        # Will be determined on first call
        self.dimension = None

    async def create(self, input_data: str | List[str]) -> List[float]:
        """
        Create embedding for a SINGLE text.

        Args:
            input_data: Single string to embed (if list, uses first element)

        Returns:
            Single embedding vector as List[float]
        """
        # EmbedderClient.create() expects single vector output
        if isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError("Empty list provided to create()")
            text = input_data[0]
        else:
            text = input_data

        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model_name,
                encoding_format="float"
            )

            embedding = response.data[0].embedding

            # Store dimension on first call
            if self.dimension is None:
                self.dimension = len(embedding)
                logger.info(f"Embedding dimension: {self.dimension}")

            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for batch of texts.

        Args:
            input_data_list: List of strings to embed

        Returns:
            List of embedding vectors
        """
        if not input_data_list:
            return []

        try:
            # The API might support batch requests
            # If it does, this is more efficient
            response = await self.client.embeddings.create(
                input=input_data_list,
                model=self.model_name,
                encoding_format="float"
            )

            embeddings = [item.embedding for item in response.data]

            # Store dimension on first call
            if self.dimension is None and embeddings:
                self.dimension = len(embeddings[0])
                logger.info(f"Embedding dimension: {self.dimension}")

            return embeddings

        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to sequential: {e}")

            # Fallback: Process one by one
            embeddings = []
            for text in input_data_list:
                embedding = await self.create(text)
                embeddings.append(embedding)

            return embeddings


# Factory function for easy switching
def get_embedder(use_hosted: bool = False) -> EmbedderClient:
    """
    Get embedder instance based on configuration.

    Args:
        use_hosted: If True, use hosted API; if False, use local sentence-transformers

    Returns:
        EmbedderClient instance
    """
    if use_hosted:
        logger.info("Using hosted Qwen embeddings")
        return HostedQwenEmbedder()
    else:
        logger.info("Using local sentence-transformers")
        from clients.graphiti_client import CustomEmbedder
        return CustomEmbedder()