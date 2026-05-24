import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_PATH = os.path.expanduser("~/Documents/supreme-court-rag/data/qdrant_store")
COLLECTION_NAME = "supreme_court"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

print("Loading model and Qdrant...")
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)
print("Ready.\n")

def retrieve(query, top_k=5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # results = client.search(
    #     collection_name=COLLECTION_NAME,
    #     query_vector=query_embedding,
    #     limit=top_k,
    #     with_payload=True
    # )
# AIzaSyAo48_JRLghWNsyz4-Whw5MOS2k7bv3X-k
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    ).points

    return results

if __name__ == "__main__":
    query = "What has the Supreme Court ruled about anticipatory bail?"
    print(f"Query: {query}\n")
    print("-" * 60)

    results = retrieve(query)

    for i, result in enumerate(results):
        print(f"\nResult {i+1}")
        print(f"Score:    {result.score:.4f}")
        print(f"Case:     {result.payload['doc_id']}")
        print(f"Year:     {result.payload['year']}")
        print(f"Chunk:    {result.payload['chunk_index']} of {result.payload['total_chunks']}")
        print(f"Text:     {result.payload['text'][:300]}...")
        print("-" * 60)