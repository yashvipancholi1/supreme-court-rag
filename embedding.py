import json
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

CHUNKS_PATH = os.path.expanduser("~/Documents/supreme-court-rag/data/chunks.json")
QDRANT_PATH = os.path.expanduser("~/Documents/supreme-court-rag/data/qdrant_store")
COLLECTION_NAME = "supreme_court"
BATCH_SIZE = 64
MODEL_NAME = "BAAI/bge-large-en-v1.5"

print("Loading embedding model (first time downloads ~1.3GB, be patient)...")
model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {EMBEDDING_DIM}")

print("Loading chunks...")
with open(CHUNKS_PATH) as f:
    chunks = json.load(f)
print(f"Total chunks to embed: {len(chunks)}")

print("Setting up Qdrant...")
client = QdrantClient(path=QDRANT_PATH)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE
    )
)
print(f"Collection '{COLLECTION_NAME}' created.")

print("Embedding and uploading chunks...")
for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i + BATCH_SIZE]
    texts = [c["text"] for c in batch]
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=embedding.tolist(),
            payload={
                "doc_id": chunk["doc_id"],
                "year": chunk["year"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "text": chunk["text"]
            }
        )
        for chunk, embedding in zip(batch, embeddings)
    ]
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print(f"\nDone! All {len(chunks)} chunks embedded and stored.")
print(f"Qdrant store saved to: {QDRANT_PATH}")

info = client.get_collection(COLLECTION_NAME)
print(f"Vectors in collection: {info.points_count}")