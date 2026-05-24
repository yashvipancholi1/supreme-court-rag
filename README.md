# Supreme Court RAG System

A production-grade Retrieval Augmented Generation (RAG) system for Indian Supreme Court judgements, demonstrating end-to-end pipeline design, semantic retrieval with diversity filtering, and rigorous evaluation methodology using RAGAS framework.

## Overview

This system provides accurate, citation-grounded answers to legal questions by retrieving relevant excerpts from 300 Supreme Court judgements (2021-2025) and generating answers that cite specific source cases. Built to demonstrate production-grade RAG engineering for AI/ML roles, the project emphasizes honest evaluation, corpus coverage analysis, and iterative prompt optimization.

**Key Features:**
- Semantic search over 20,469 embedded legal text chunks
- Diversity filtering preventing single-document dominance in results
- Citation-grounded generation requiring source attribution
- Honest admission when corpus lacks sufficient information
- RAGAS-based evaluation with manually curated ground truth
- Documented prompt optimization improving answer relevancy by 40%

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│  PDFs → Text Extraction → Semantic Chunking → Embedding    │
│         (PyMuPDF)         (1000 chars/200 overlap)          │
│                                  ↓                           │
│                         Qdrant Vector Store                 │
│                         (20,469 chunks)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  Question → Semantic Search (top-20) → Diversity Filter    │
│                                         (max 2/doc)         │
│                    ↓                                        │
│         Context Builder → LLM Generation → Cited Answer     │
│         (source metadata)  (GPT-4o-mini)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  18-question benchmark → RAGAS metrics → Performance scores │
│  (manual ground truth)   (OpenAI judge)                     │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

**Core Pipeline:**
- **Embeddings:** `BAAI/bge-large-en-v1.5` via HuggingFace Transformers (free, local)
- **Vector Store:** Qdrant (local mode)
- **Generation:** GPT-4o-mini via OpenAI API
- **Evaluation:** RAGAS framework with OpenAI-based LLM judge

**Data Processing:**
- PyMuPDF for PDF text extraction
- Sentence Transformers for semantic embeddings
- Python 3.12, Pandas, NumPy

**Infrastructure:**
- 300 Supreme Court judgement PDFs (2021-2025)
- 20,469 semantic chunks with overlap
- SQLite logging (optional, for production extension)

## Evaluation Results

### Baseline (Verbose Generation)
Initial system with comprehensive legal context generation:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.94 |
| Answer Relevancy | **0.64** |
| Context Precision | 0.85 |
| Context Recall | 0.97 |

### After Prompt Optimization
Added conciseness constraint to generation prompt:

| Metric | Score | Change |
|--------|-------|--------|
| Faithfulness | 0.94 | — |
| Answer Relevancy | **0.90** | **+0.26** |
| Context Precision | 0.86 | +0.01 |
| Context Recall | 0.95 | -0.02 |

**Key Findings:**
- **94% Faithfulness:** Answers almost never contradict source documents
- **90% Answer Relevancy:** Improved from 64% through prompt engineering
- **95% Context Recall:** Near-perfect retrieval coverage when relevant judgements exist in corpus
- **Corpus Coverage:** 75% of test questions received proper answers; 25% correctly identified as having insufficient source material

### Evaluation Methodology

18-question legal benchmark with manually written ground truth answers verified against retrieved source chunks. Questions span criminal law, constitutional rights, family law, civil disputes, and procedural law. Evaluation performed using RAGAS framework with GPT-4o-mini as LLM judge, measuring:

- **Faithfulness:** Do generated answers contradict retrieved sources?
- **Answer Relevancy:** Do answers directly address questions asked?
- **Context Precision:** What fraction of retrieved chunks were actually useful?
- **Context Recall:** Did retrieval find all necessary information?

## Setup Instructions

### Prerequisites
- Python 3.10+
- 16GB RAM (for local embeddings)
- 5GB free disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/supreme-court-rag.git
cd supreme-court-rag
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
```

### Download and Prepare Data

1. **Download Supreme Court judgements dataset:**

Option A - Kaggle (recommended):
```bash
# Install Kaggle CLI and authenticate
pip install kaggle
# Download dataset
kaggle datasets download adarshsingh0903/legal-dataset-sc-judgments-india-19502024
```

Option B - Manual download from Kaggle website

2. **Sample and process documents:**
```bash
python scripts/sample_pdfs.py    # Samples 300 PDFs (60 per year, 2021-2025)
python scripts/ingest.py          # Extracts text from PDFs
python scripts/chunker.py         # Creates semantic chunks
```

3. **Build vector index:**
```bash
python scripts/embed.py           # Embeds chunks and creates Qdrant index
# First run downloads ~1.3GB embedding model (cached afterwards)
# Takes ~20-30 minutes for 20,469 chunks
```

### Run the System

**Interactive query:**
```bash
python pipeline/generate.py
```

**Run evaluation:**
```bash
python pipeline/eval_ragas.py
# Requires eval_dataset.json with ground truth answers
# Costs ~$0.08 in OpenAI API calls
```

## Project Structure

```
supreme-court-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                    # Downloaded PDFs (gitignored)
│   ├── sampled/                # 300 selected PDFs (gitignored)
│   ├── extracted.json          # Text extracted from PDFs (gitignored)
│   ├── chunks.json             # Semantic chunks (gitignored)
│   ├── qdrant_store/           # Vector database (gitignored)
│   ├── eval_dataset.json       # Ground truth Q&A pairs (tracked)
│   └── eval_results.json       # RAGAS scores (tracked)
│
├── scripts/
│   ├── sample_pdfs.py          # Random sampling from full dataset
│   ├── ingest.py               # PDF → text extraction
│   ├── chunker.py              # Text → semantic chunks
│   └── embed.py                # Chunks → vector embeddings
│
└── pipeline/
    ├── generate.py             # Query → retrieval → generation
    └── eval_ragas.py           # RAGAS evaluation runner
```

## Reproducibility

All results are reproducible given:
1. Same 300-document sample (deterministic seed in `sample_pdfs.py`)
2. Same embedding model (`BAAI/bge-large-en-v1.5`)
3. Same chunking parameters (1000/200)
4. Same evaluation dataset (`data/eval_dataset.json` in repo)

Eval scores may vary by ±0.02 due to OpenAI API non-determinism despite temperature=0.

## License

MIT License - see LICENSE file for details.

Dataset source: [Kaggle - Legal Dataset SC Judgments India](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024)

## Acknowledgments

- RAGAS framework for evaluation methodology
- Qdrant for vector database
- HuggingFace for embedding models
- OpenAI for generation API

---

**Author:** Yashvi
**Contact:** pancholiyashvi123@gmail.com
**LinkedIn:** https://www.linkedin.com/in/yashvi-pancholi/

---

*Built as a technical portfolio project demonstrating production RAG system design, evaluation-driven development, and MLOps best practices for AI/ML engineering roles.*