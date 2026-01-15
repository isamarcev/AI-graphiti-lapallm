import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from config.settings import settings

logger = logging.getLogger(__name__)


_DISTANCE_MAP = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
}


class QdrantClient:
    """
    Minimal Qdrant client wrapper with insert and similarity search.
    """

    def __init__(
        self,
        collection: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
    ):
        self.collection = collection or settings.qdrant_collection
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.vector_size = vector_size or settings.embedding_dimension
        self.distance = distance or settings.qdrant_distance
        self._client: Optional[AsyncQdrantClient] = None

    async def initialize(self) -> "QdrantClient":
        """
        Create underlying client and ensure the collection exists.
        """
        if self._client is not None:
            return self

        distance = _DISTANCE_MAP.get(self.distance.lower())
        if distance is None:
            raise ValueError(f"Unsupported distance metric: {self.distance}")

        self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key)

        try:
            info = await self._client.get_collection(self.collection)
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_size:
                raise ValueError(
                    f"Existing collection '{self.collection}' has vector size "
                    f"{existing_size}, expected {self.vector_size}. "
                    "Recreate the collection with the correct dimension."
                )
            logger.info("Qdrant collection '%s' already exists", self.collection)
        except Exception:
            logger.info(
                "Creating Qdrant collection '%s' (size=%s, distance=%s)",
                self.collection,
                self.vector_size,
                distance,
            )
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=distance,
                ),
            )

        return self

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    def _to_point_id(self, record_id: Optional[str]) -> uuid.UUID:
        """
        Convert arbitrary string IDs to a deterministic UUID accepted by Qdrant.
        If None provided, generate a random UUID4.
        """
        if record_id is None:
            return uuid.uuid4()
        return uuid.uuid5(uuid.NAMESPACE_URL, record_id)

    async def insert_record(
        self,
        record_id: Optional[str],
        vector: List[float],
        fact: str,
        message_id: str,
        is_relevant: bool,
    ):
        """
        Insert a single record. The provided record_id (string) is stored in payload
        and converted to a Qdrant-compatible UUID for the point ID.
        """
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized. Call initialize() first.")

        point_id = self._to_point_id(record_id)
        payload = {
            "fact": fact,
            "messageid": message_id,
            "is_relevant": is_relevant,
            "record_id": record_id or str(point_id),
        }

        await self._client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def set_relevance_by_message_id(self, message_id: str, is_relevant: bool) -> None:
        """
        Update is_relevant payload flag for all points matching the given messageid.
        """
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized. Call initialize() first.")

        filter_condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="messageid",
                    match=qmodels.MatchValue(value=message_id),
                )
            ]
        )

        await self._client.set_payload(
            collection_name=self.collection,
            payload={"is_relevant": is_relevant},
            filter=filter_condition,
        )

    async def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        only_relevant: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search similar vectors, optionally filtering by is_relevant==True.
        Returns list of dicts with id, score, payload (vector omitted).
        """
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized. Call initialize() first.")

        search_filter = None
        if only_relevant:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="is_relevant",
                        match=qmodels.MatchValue(value=True),
                    )
                ]
            )

        
        results = await self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=search_filter,
        )

        hits = results.points if hasattr(results, "points") else results

        mapped_results: List[Dict[str, Any]] = []
        for hit in hits:
            # qdrant-client 1.16.x returns ScoredPoint; defensive tuple/list handling included
            if hasattr(hit, "id") and hasattr(hit, "score"):
                mapped_results.append(
                    {"id": hit.id, "score": hit.score, "payload": getattr(hit, "payload", None)}
                )
            elif isinstance(hit, (list, tuple)) and len(hit) >= 2:
                mapped_results.append({"id": hit[0], "score": hit[1], "payload": None})

        return mapped_results
