from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from volcenginesdkarkruntime import Ark
import os
from urllib.parse import unquote
from urllib.parse import urlparse
from dotenv import load_dotenv
import json

load_dotenv()




DEFAULT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "your_collection_name")
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "your_qdrant_endpoint")
DEFAULT_EMBEDDING_MODEL = os.environ.get("ARK_EMBEDDING_MODEL", "multimodal_embedding_name")
DEFAULT_LIMIT = int(os.environ.get("QDRANT_LIMIT", "1"))
MAX_IMAGE_WIDTH_PX = int(os.environ.get("TERMINAL_IMAGE_MAX_WIDTH_PX", "256"))
IMAGE_URL_LIST = json.loads(os.environ.get("IMAGE_URL_LIST", "[]"))

collection_name =DEFAULT_COLLECTION_NAME

qdrant_client = QdrantClient(url=DEFAULT_QDRANT_URL)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
)


client = Ark(
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key=os.environ.get("ARK_API_KEY"),
)


image_url_list = IMAGE_URL_LIST


pts = []
for idx, image_url in enumerate(image_url_list, start=1):
    # print(image_url)
    parsed_url = urlparse(image_url)
    filename = os.path.basename(parsed_url.path)
    decoded_filename = unquote(filename)

    resp = client.multimodal_embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=[
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
        ]
    )
    
    pts.append(
        PointStruct(
            id=idx,
            vector=resp.data.embedding,
            payload={
                "file_name": decoded_filename,
                "image_url": image_url
            }
        )
    )

operation_info = qdrant_client.upsert(
    collection_name=collection_name,
    wait=True,
    points=pts,
)

print(operation_info)
