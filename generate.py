import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

load_dotenv()

QDRANT_PATH = os.path.expanduser("~/Documents/supreme-court-rag/data/qdrant_store")
COLLECTION_NAME = "supreme_court"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

print("Loading model and Qdrant...")
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Ready.\n")


def retrieve(query, top_k=5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    ).points

    return results


#To add diversity: doesn't give answer from just one document that it finds most relvant
def diversify(results, max_per_doc=2):
    seen = {}
    diversified = []
    
    for result in results:
        doc_id = result.payload["doc_id"]
        if seen.get(doc_id, 0) < max_per_doc:
            diversified.append(result)
            seen[doc_id] = seen.get(doc_id, 0) + 1
    
    return diversified

def build_context(results):
    context_parts = []
    for i, result in enumerate(results):
        context_parts.append(
            f"[Source {i+1}]\n"
            f"Case: {result.payload['doc_id']}\n"
            f"Year: {result.payload['year']}\n"
            f"Text: {result.payload['text']}\n"
        )
    return "\n---\n".join(context_parts)

def generate_answer(query, context):
    prompt = f"""You are a legal research assistant specializing in Indian Supreme Court judgements.

Answer the following question using ONLY the provided source excerpts from Supreme Court judgements.
For every point you make, cite the source number it came from using [Source X] notation.
If the sources do not contain enough information to answer the question, say so honestly.
Do not use any knowledge outside of the provided sources.

Answer concisely and directly. Focus only on what the question asks. Avoid tangential legal background unless directly relevant to answering the specific question.

Question: {query}

Sources:
{context}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content




def ask(query):
    print(f"Query: {query}\n")
    # print("Retrieving relevant chunks...")
    results = retrieve(query, top_k=20)
    results = diversify(results, max_per_doc=2)
    
    # print(f"Found {len(results)} relevant chunks\n")
    # print("Sources used:")
    # for i, r in enumerate(results):
        # print(f"  [{i+1}] {r.payload['doc_id']} (score: {r.score:.4f})")
    
    # print("\nGenerating answer...\n")
    context = build_context(results)
    answer = generate_answer(query, context)
    
    # print("=" * 60)
    print("ANSWER:")
    # print("=" * 60)
    print(answer)
    # print("=" * 60)

if __name__ == "__main__":
    ask("What conditions has the Supreme Court set for granting bail in serious criminal cases?")