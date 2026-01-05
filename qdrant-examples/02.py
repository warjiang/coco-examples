from qdrant_client import QdrantClient

from dotenv import load_dotenv
import os

load_dotenv()


DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "your_qdrant_endpoint")
client = QdrantClient(url=DEFAULT_QDRANT_URL)


search_result = client.query_points(
    collection_name="test_collection",
    query=[0.2, 0.1, 0.9, 0.7],
    # query_filter=Filter(
    #     must=[FieldCondition(key="city", match=MatchValue(value="London"))]
    # ),
    with_payload=True,
    limit=3
).points

# print(search_result)
for point in search_result:
    print(point)