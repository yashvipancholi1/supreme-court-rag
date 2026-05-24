import json
import os

input_path = os.path.expanduser("~/Documents/supreme-court-rag/data/extracted.json")
output_path = os.path.expanduser("~/Documents/supreme-court-rag/data/chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

with open(input_path) as f:
    documents = json.load(f)

all_chunks = []
chunk_id = 0

for doc in documents:
    text = doc["text"]
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc["id"],
            "year": doc["year"],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk
        })
        chunk_id += 1

    print(f"Chunked: {doc['id']} → {len(chunks)} chunks")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"\nTotal chunks created: {len(all_chunks)}")
print(f"Saved to: {output_path}")