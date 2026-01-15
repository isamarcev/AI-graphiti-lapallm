# save as test_qdrant.py
import asyncio
from clients.qdrant_client import QdrantClient

async def main():
    qc = QdrantClient(
        url="http://localhost:6333",   # host access
        collection="facts_test",            # or override
    )
    await qc.initialize()

    await qc.insert_record(
        record_id="test-3",
        vector=[0.55] * 4096,            # match settings.embedding_dimension
        fact="sabakas",
        message_id="m-905",
        is_relevant=True,
    )
    await qc.insert_record(
        record_id="test-2",
        vector=[0.73] * 4096,            # match settings.embedding_dimension
        fact="sabakasss ne ta",
        message_id="m-35000",
        is_relevant=True,
    )

    results = await qc.search_similar(
        query_vector=[0.8] * 4096,
        top_k=3,
        only_relevant=True,
    )
    print(results)

    await qc.close()

asyncio.run(main())