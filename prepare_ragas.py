import os
import re
import json

BASE_PATH = os.path.expanduser("~/Documents/supreme-court-rag/data")

GROUND_TRUTH_PATH = os.path.join(BASE_PATH, "eval_dataset.json")
ANSWERS_PATH = os.path.join(BASE_PATH, "question_answers.txt")
CHUNKS_PATH = os.path.join(BASE_PATH, "question_chunks.txt")

OUTPUT_PATH = os.path.join(BASE_PATH, "generated_answers.json")


# LOAD GROUND TRUTH
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

ground_truth_map = {
    item["question"].strip(): item["ground_truth"].strip()
    for item in ground_truth_data
}


# LOAD ANSWERS FILE
with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
    answers_text = f.read()


# LOAD CHUNKS FILE
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_text = f.read()


# -------------------------------
# EXTRACT QUESTIONS + ANSWERS
# -------------------------------

qa_pattern = re.compile(
    r"QUESTION:\s*(.*?)\n=+\n.*?ANSWER:\n(.*?)(?=\n=+\nTIMESTAMP:|\Z)",
    re.DOTALL
)

qa_matches = qa_pattern.findall(answers_text)

answers_map = {}

for question, answer in qa_matches:
    question = question.strip()
    answer = answer.strip()

    answers_map[question] = answer


# -------------------------------
# EXTRACT QUESTIONS + CONTEXTS
# -------------------------------

question_block_pattern = re.compile(
    r"QUESTION:\s*(.*?)\n=+\n(.*?)(?=\n=+\nTIMESTAMP:|\Z)",
    re.DOTALL
)

question_blocks = question_block_pattern.findall(chunks_text)

contexts_map = {}

for question, block in question_blocks:

    question = question.strip()

    context_chunks = []

    chunk_pattern = re.compile(
        r"Text:\n(.*?)(?=\n-+\n|\Z)",
        re.DOTALL
    )

    chunk_matches = chunk_pattern.findall(block)

    for chunk in chunk_matches:
        cleaned_chunk = chunk.strip()

        if cleaned_chunk:
            context_chunks.append(cleaned_chunk)

    contexts_map[question] = context_chunks


# -------------------------------
# MERGE EVERYTHING
# -------------------------------

final_data = []

all_questions = set(ground_truth_map.keys())

for question in all_questions:

    answer = answers_map.get(question)

    contexts = contexts_map.get(question)

    ground_truth = ground_truth_map.get(question)

    if not answer:
        print(f"Missing answer for question:\n{question}\n")
        continue

    if not contexts:
        print(f"Missing contexts for question:\n{question}\n")
        continue

    final_data.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth
    })


# SAVE OUTPUT
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(final_data)} entries to:")
print(OUTPUT_PATH)