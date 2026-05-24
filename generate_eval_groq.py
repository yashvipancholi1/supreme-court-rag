import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
from datetime import datetime
import time

load_dotenv()

QDRANT_PATH = os.path.expanduser("~/Documents/supreme-court-rag/data/qdrant_store")
COLLECTION_NAME = "supreme_court"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHUNKS_LOG = os.path.expanduser("~/Documents/supreme-court-rag/data/question_chunks_2.txt")
ANSWERS_LOG = os.path.expanduser("~/Documents/supreme-court-rag/data/question_answers_2.txt")

print("Loading model and Qdrant...")
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path=QDRANT_PATH)
# from openai import OpenAI
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Ready.\n")


def retrieve(query, top_k=20):
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
Do not use any knowledge outside of the provided sources. Answer concisely and directly. Focus only on what the question asks. 
Avoid tangential legal background unless directly relevant to answering the specific question.

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


def log_chunks(query, results):
    with open(CHUNKS_LOG, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"QUESTION: {query}\n")
        f.write("=" * 80 + "\n")
        for i, r in enumerate(results):
            f.write(f"\n[Source {i+1}]\n")
            f.write(f"Case: {r.payload['doc_id']}\n")
            f.write(f"Year: {r.payload['year']}\n")
            f.write(f"Score: {r.score:.4f}\n")
            f.write(f"Chunk index: {r.payload['chunk_index']} of {r.payload['total_chunks']}\n")
            f.write(f"Text:\n{r.payload['text']}\n")
            f.write("-" * 40 + "\n")


def log_answer(query, results, answer):
    with open(ANSWERS_LOG, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"QUESTION: {query}\n")
        f.write("=" * 80 + "\n")
        f.write("\nSOURCES USED:\n")
        for i, r in enumerate(results):
            f.write(f"  [{i+1}] {r.payload['doc_id']} (score: {r.score:.4f})\n")
        f.write(f"\nANSWER:\n{answer}\n")


def ask(query):
    print(f"Query: {query}\n")

    results = retrieve(query, top_k=20)
    results = diversify(results, max_per_doc=2)

    context = build_context(results)
    answer = generate_answer(query, context)

    log_chunks(query, results)
    log_answer(query, results, answer)

    print("ANSWER:")
    print(answer)
    print(f"\n[Logged to {CHUNKS_LOG}]")
    print(f"[Logged to {ANSWERS_LOG}]")


if __name__ == "__main__":
    questions = [
        # Criminal Law
        "What conditions has the Supreme Court set for granting bail in cases involving economic offences?",
        "How has the Supreme Court ruled on the cancellation of bail after it has been granted?",
        "What has the Supreme Court said about the rights of accused persons during police interrogation?",
        "How has the court ruled on the use of circumstantial evidence in murder cases?",
        "What are the Supreme Court guidelines on sentencing in cases of rape and sexual assault?",

        # Constitutional Law
        "How has the Supreme Court interpreted the right to equality under Article 14?",
        "What has the court ruled about reservation policies in government jobs and promotions?",
        "How has the Supreme Court ruled on the validity of preventive detention?",
        "What has the court said about the fundamental right to education under Article 21A?",
        "How has the Supreme Court interpreted the separation of powers between judiciary and executive?",

        # Family and Personal Law
        "What grounds has the Supreme Court recognized for granting divorce on irretrievable breakdown of marriage?",
        "How has the court ruled on maintenance rights of divorced Muslim women?",
        "What has the Supreme Court ruled about adoption rights and procedures?",
        "How has the court ruled on succession rights of daughters in Hindu joint family property?",
        "What has the Supreme Court said about domestic violence and protection orders?",

        # Civil and Commercial Law
        "How has the Supreme Court interpreted the doctrine of promissory estoppel in contract disputes?",
        "What has the court ruled about specific performance of contracts in property transactions?",
        "How has the Supreme Court ruled on arbitration awards and grounds for their challenge?",
        "What has the court said about the liability of directors in cases of company fraud?",
        "How has the court ruled on consumer protection and deficiency of service claims?",

        # Procedural Law
        "What has the Supreme Court ruled about the powers of High Courts under Article 226?",
        "How has the court interpreted the limitation period for filing civil suits?",
        "What has the Supreme Court said about the admissibility of electronic evidence in court?",
        "How has the court ruled on the scope of judicial review of administrative decisions?",
        "What has the Supreme Court ruled about the right of appeal and its limitations in criminal cases?",
    ]

    for q in questions:
        ask(q)
        time.sleep(10)