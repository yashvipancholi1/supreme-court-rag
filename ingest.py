import fitz
import os
import json
import re

source_base = os.path.expanduser("~/Documents/supreme-court-rag/data/sampled")
output_path = os.path.expanduser("~/Documents/supreme-court-rag/data/extracted.json")

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'IN THE SUPREME COURT OF INDIA', '', text)
    text = text.strip()
    return text

def extract_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return clean_text(full_text)

documents = []
failed = []

years = ["2021", "2022", "2023", "2024", "2025"]

for year in years:
    year_folder = os.path.join(source_base, year)
    pdfs = [f for f in os.listdir(year_folder) if f.endswith(".PDF")]

    for pdf_name in pdfs:
        path = os.path.join(year_folder, pdf_name)
        try:
            text = extract_pdf(path)
            if len(text) < 200:
                print(f"Skipping (too short): {pdf_name}")
                continue
            documents.append({
                "id": pdf_name.replace(".pdf", ""),
                "year": year,
                "source": path,
                "text": text
            })
            print(f"Extracted: {pdf_name} ({len(text)} chars)")
        except Exception as e:
            print(f"Failed: {pdf_name} — {e}")
            failed.append(pdf_name)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"\nDone.")
print(f"Successfully extracted: {len(documents)} documents")
print(f"Failed: {len(failed)} documents")
print(f"Saved to: {output_path}")