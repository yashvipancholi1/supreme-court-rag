import os
import json
import math
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# INPUT FILE
GENERATED_ANSWERS_PATH = os.path.expanduser(
    "~/Documents/supreme-court-rag/data/generated_answers.json"
)

# OUTPUT FILE
RESULTS_PATH = os.path.expanduser(
    "~/Documents/supreme-court-rag/data/eval_results_2.json"
)

print("Loading generated answers...\n")

with open(GENERATED_ANSWERS_PATH, "r", encoding="utf-8") as f:
    generated_data = json.load(f)

# REMOVE INVALID ENTRIES
cleaned_data = []

for item in generated_data:

    if (
        item.get("question")
        and item.get("answer")
        and item.get("contexts")
        and item.get("ground_truth")
    ):
        cleaned_data.append(item)

generated_data = cleaned_data

print(f"Loaded {len(generated_data)} questions for evaluation\n")

# PREPARE DATASET
questions = [x["question"] for x in generated_data]
answers = [x["answer"] for x in generated_data]
contexts = [x["contexts"] for x in generated_data]
ground_truths = [x["ground_truth"] for x in generated_data]

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print("Initializing RAGAS models...\n")

# LLM
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0
    )
)

# EMBEDDINGS
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
)

print("=" * 80)
print("Running RAGAS evaluation...")
print("=" * 80)

try:

    # RUN EVALUATION
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 80)

    # CONVERT TO DATAFRAME
    df = results.to_pandas()

    print("\nAvailable columns in results:")
    print(df.columns.tolist())

    # SAFE METRIC EXTRACTION
    def get_score(df, key):

        try:

            if key not in df.columns:
                return 0.0

            vals = df[key].dropna().tolist()

            vals = [
                float(v)
                for v in vals
                if v is not None
                and not (
                    isinstance(v, float)
                    and math.isnan(v)
                )
            ]

            if len(vals) == 0:
                return 0.0

            return sum(vals) / len(vals)

        except Exception as e:
            print(f"Error extracting {key}: {e}")
            return 0.0

    # SAFE FLOAT
    def safe_float(v):

        try:

            if v is None:
                return None

            v = float(v)

            if math.isnan(v):
                return None

            return v

        except:
            return None

    # OVERALL SCORES
    faithfulness_score = get_score(df, "faithfulness")
    relevancy_score = get_score(df, "answer_relevancy")
    precision_score = get_score(df, "context_precision")
    recall_score = get_score(df, "context_recall")

    # PRINT RESULTS
    print(f"Faithfulness:        {faithfulness_score:.4f}")
    print(f"Answer Relevancy:    {relevancy_score:.4f}")
    print(f"Context Precision:   {precision_score:.4f}")
    print(f"Context Recall:      {recall_score:.4f}")

    print("=" * 80)

    # RESULTS JSON
    results_dict = {
        "summary": {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "context_precision": precision_score,
            "context_recall": recall_score
        },
        "per_question": []
    }

    # PER QUESTION METRICS
    for idx, row in df.iterrows():

        question = questions[idx] if idx < len(questions) else "Unknown"

        results_dict["per_question"].append({
            "question": question,
            "faithfulness": safe_float(row.get("faithfulness")),
            "answer_relevancy": safe_float(row.get("answer_relevancy")),
            "context_precision": safe_float(row.get("context_precision")),
            "context_recall": safe_float(row.get("context_recall"))
        })

    # SAVE RESULTS
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to:")
    print(RESULTS_PATH)

except Exception as e:

    print("\nERROR DURING EVALUATION")
    print(str(e))

print("\nEvaluation complete!")