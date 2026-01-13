"""
Hosted reranker using Lapathon/vLLM API for passage ranking.
Alternative to local BGE cross-encoder with better performance.
"""

from typing import List, Tuple
import asyncio
import logging
import time
from openai import AsyncOpenAI
import httpx

# Try to import the base class
try:
    from graphiti_core.cross_encoder import CrossEncoderClient
except ImportError:
    from graphiti_core.cross_encoder.client import CrossEncoderClient

from config.settings import settings

logger = logging.getLogger(__name__)


class HostedRerankerClient(CrossEncoderClient):
    """
    Reranker using hosted LLM API (Lapathon/vLLM).

    This reranker uses the hosted LLM to score passage relevance
    instead of local cross-encoder models. It's faster than BGE on CPU
    but slower than NoOp reranker.

    Requirements:
    - API must support logprobs (for probabilistic scoring)
    - OR fallback to simple yes/no classification

    Performance:
    - Faster than BGE cross-encoder on CPU
    - Slower than NoOp (makes N API calls)
    - Network latency dependent
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model_name: str = None,
        use_logprobs: bool = True,
        max_concurrent: int = 5
    ):
        """
        Initialize hosted reranker.

        Args:
            base_url: API base URL (default from settings)
            api_key: API key (default from settings)
            model_name: Model name (default from settings)
            use_logprobs: Try to use logprobs for scoring (requires API support)
            max_concurrent: Maximum concurrent API requests
        """
        # Strip /v1 suffix if present
        self.base_url = (base_url or settings.vllm_base_url).rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'

        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.vllm_model_name
        self.use_logprobs = use_logprobs
        self.max_concurrent = max_concurrent

        logger.info(f"Initializing hosted reranker: {self.model_name}")
        logger.info(f"API URL: {self.base_url}")
        logger.info(f"Use logprobs: {use_logprobs}")
        logger.info(f"Max concurrent: {max_concurrent}")

        # Create httpx client with increased limits for true parallelism
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_concurrent * 2,  # Total connections
                max_keepalive_connections=max_concurrent,  # Keep-alive pool
            ),
            timeout=httpx.Timeout(30.0, connect=10.0)
        )

        # Create OpenAI client with custom http client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client
        )

        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _score_passage_logprobs(
        self,
        query: str,
        passage: str
    ) -> float:
        """
        Score passage using logprobs (if API supports it).

        Args:
            query: Search query
            passage: Passage to score

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Prompt asking for relevance classification
        system_message = "You are a relevance classifier. Answer ONLY with 'Yes' or 'No'."
        user_message = f"""Is the following passage relevant to the query?

Query: {query}

Passage: {passage}

Answer (Yes/No):"""

        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0,
                    max_tokens=5,
                    logprobs=True,  # Request log probabilities
                    top_logprobs=2  # Get top 2 token probabilities
                )

                # Extract logprobs for "Yes" and "No" tokens
                # NOTE: This is vLLM-specific, may need adjustment
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    token_logprobs = response.choices[0].logprobs.content[0]

                    # Try to find Yes/No tokens
                    yes_logprob = None
                    no_logprob = None

                    for top_logprob in token_logprobs.top_logprobs:
                        token = top_logprob.token.lower().strip()
                        if 'yes' in token or 'так' in token:
                            yes_logprob = top_logprob.logprob
                        elif 'no' in token or 'ні' in token:
                            no_logprob = top_logprob.logprob

                    # Calculate score from logprobs
                    if yes_logprob is not None and no_logprob is not None:
                        # Convert to probabilities
                        import math
                        yes_prob = math.exp(yes_logprob)
                        no_prob = math.exp(no_logprob)
                        score = yes_prob / (yes_prob + no_prob)
                        return score

                # Fallback: simple text classification
                answer = response.choices[0].message.content.strip().lower()
                if 'yes' in answer or 'так' in answer:
                    return 0.9
                elif 'no' in answer or 'ні' in answer:
                    return 0.1
                else:
                    return 0.5

        except Exception as e:
            logger.warning(f"Error scoring passage with logprobs: {e}")
            return 0.5  # Default neutral score

    async def _score_passage_simple(
        self,
        query: str,
        passage: str
    ) -> float:
        """
        Score passage using simple yes/no classification.

        Args:
            query: Search query
            passage: Passage to score

        Returns:
            Relevance score (0.0 to 1.0)
        """
        system_message = "You are a relevance classifier. Answer ONLY with 'Yes' or 'No'."
        user_message = f"""Is the following passage relevant to the query?

Query: {query}

Passage: {passage}

Answer (Yes/No):"""

        try:
            async with self.semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0,
                    max_tokens=5
                )

                answer = response.choices[0].message.content.strip().lower()

                # Simple scoring based on answer
                if 'yes' in answer or 'так' in answer:
                    return 0.9
                elif 'no' in answer or 'ні' in answer:
                    return 0.2
                else:
                    return 0.5

        except Exception as e:
            logger.warning(f"Error scoring passage: {e}")
            return 0.5

    async def rank(
        self,
        query: str,
        passages: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rank passages by relevance to query using hosted LLM.

        Args:
            query: Search query
            passages: List of passages to rank

        Returns:
            List of (passage, score) tuples sorted by relevance (highest first)
        """
        if not passages:
            return []

        start_time = time.time()
        logger.info(f"Reranking {len(passages)} passages with hosted API (max_concurrent={self.max_concurrent})")

        # Choose scoring method based on settings
        score_fn = (
            self._score_passage_logprobs
            if self.use_logprobs
            else self._score_passage_simple
        )

        # Score all passages concurrently
        scoring_start = time.time()
        tasks = [score_fn(query, passage) for passage in passages]
        scores = await asyncio.gather(*tasks)
        scoring_time = time.time() - scoring_start

        # Combine passages with scores
        results = list(zip(passages, scores))

        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        total_time = time.time() - start_time
        avg_per_passage = scoring_time / len(passages) if passages else 0
        logger.info(f"Reranking complete in {total_time:.3f}s (scoring: {scoring_time:.3f}s, avg: {avg_per_passage:.3f}s/passage)")
        logger.info(f"Top score: {results[0][1]:.3f}")

        return results