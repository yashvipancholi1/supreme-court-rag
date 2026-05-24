import os
import random
import shutil

source_base = os.path.expanduser("~/Documents/supreme-court-rag/data/supreme_court_judgments")
dest_base = os.path.expanduser("~/Documents/supreme-court-rag/data/sampled")
years = ["2021", "2022", "2023", "2024", "2025"]
samples_per_year = 60

os.makedirs(dest_base, exist_ok=True)

total_copied = 0

for year in years:
    year_folder = os.path.join(source_base, year)
    dest_folder = os.path.join(dest_base, year)
    os.makedirs(dest_folder, exist_ok=True)

    pdfs = [f for f in os.listdir(year_folder) if f.endswith(".PDF")]
    sampled = random.sample(pdfs, min(samples_per_year, len(pdfs)))

    for pdf in sampled:
        src = os.path.join(year_folder, pdf)
        dst = os.path.join(dest_folder, pdf)
        shutil.copy2(src, dst)
        total_copied += 1

    print(f"{year}: copied {len(sampled)} PDFs")

print(f"\nTotal sampled: {total_copied} PDFs")
print(f"Saved to: {dest_base}")