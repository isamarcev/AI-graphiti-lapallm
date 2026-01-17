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
        vector: List[float],
        fact: str,
        message_id: str,
        is_relevant: bool,
        payload: Optional[Dict[str, Any]] = None,
    ):
        """
        Insert a single record. Generates a new UUID4 point id and stores metadata in payload.
        """
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized. Call initialize() first.")

        point_id = uuid.uuid4()
        base_payload = {
            "fact": fact,
            "message_id": message_id,
            "is_relevant": is_relevant,
        }
        if payload:
            base_payload.update(payload)

        await self._client.upsert(
            collection_name=self.collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=base_payload,
                )
            ],
        )

    async def set_relevance_by_message_id(self, record_id: str, is_relevant: bool) -> None:

        if self._client is None:
            raise RuntimeError("QdrantClient not initialized. Call initialize() first.")

        filter_condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="record_id",
                    match=qmodels.MatchValue(value=record_id),
                )
            ]
        )

        # Scroll to find matching point IDs
        scroll_result = await self._client.scroll(
            collection_name=self.collection,
            scroll_filter=filter_condition,
            limit=100,
            with_payload=False,
            with_vectors=False,
        )

        point_ids = [point.id for point in scroll_result[0]] if scroll_result else []
        if not point_ids:
            logger.warning("No Qdrant points found for messageid=%s", message_id)
            return

        await self._client.set_payload(
            collection_name=self.collection,
            payload={"is_relevant": is_relevant},
            points=point_ids,
        )

    async def set_relevance_by_record_id(self, record_id: str, is_relevant: bool) -> None:
        """Update is_relevant for a point using its Qdrant point ID directly."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized. Call initialize() first.")

        try:
            # Convert string record_id to UUID (assuming it's already a valid UUID string from Qdrant)
            point_id = uuid.UUID(record_id)
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid UUID format for record_id={record_id}: {e}")
            return

        await self._client.set_payload(
            collection_name=self.collection,
            payload={"is_relevant": is_relevant},
            points=[point_id],
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
