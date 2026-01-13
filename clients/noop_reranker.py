"""
No-operation reranker that bypasses reranking for performance.
Returns passages in original order without scoring.
"""

from typing import List, Tuple
import logging

# Try to import the base class
try:
    from graphiti_core.cross_encoder import CrossEncoderClient
except ImportError:
    # Fallback if module structure is different
    from graphiti_core.cross_encoder.client import CrossEncoderClient

logger = logging.getLogger(__name__)


class NoOpRerankerClient(CrossEncoderClient):
    """
    No-operation reranker that skips reranking for performance.

    This reranker simply returns passages in their original order
    with a default score, avoiding expensive cross-encoder inference.

    Use this when:
    - CPU performance is critical
    - Initial retrieval quality is sufficient
    - Cross-encoder overhead is unacceptable
    """

    def __init__(self):
        """Initialize no-op reranker."""
        logger.info("🚀 NoOp reranker initialized (no reranking, maximum speed)")

    async def rank(
        self,
        query: str,
        passages: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Return passages in original order without reranking.

        Args:
            query: Search query (ignored)
            passages: List of passages to rank

        Returns:
            List of (passage, score) tuples in original order
            Scores decrease linearly from 1.0 to 0.5
        """
        # Return passages with decreasing scores based on position
        # This maintains original order while providing reasonable scores
        num_passages = len(passages)

        if num_passages == 0:
            return []

        # Generate linearly decreasing scores: 1.0, 0.95, 0.90, ... down to 0.5
        results = []
        for i, passage in enumerate(passages):
            # Score from 1.0 to 0.5 based on position
            score = 1.0 - (i / (num_passages * 2))
            score = max(score, 0.5)  # minimum 0.5
            results.append((passage, score))

        return results