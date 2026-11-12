# Research & Knowledge Assistant

An AI-powered document intelligence system built using **LangChain and Retrieval-Augmented Generation (RAG)**.

The application will ingest research documents, retrieve relevant evidence, and generate grounded answers with document and page-level citations. It will also support research summarization, comparison, and evidence-based abstention.

## Project Objectives

- Ingest and process multiple research documents.
- Retrieve relevant passages using semantic search.
- Generate answers grounded in retrieved evidence.
- Cite supporting documents and original PDF pages.
- Summarize and compare research findings.
- Abstain when retrieved evidence is insufficient.
- Evaluate and optimize retrieval and generation independently.

## Architecture

The system follows a modular architecture with separate ingestion, retrieval, and generation components.

```text
Research Documents
       |
       v
Ingestion Pipeline
       |
       +--> PDF Loading
       +--> Document Hashing
       +--> Duplicate Detection
       +--> Text Splitting
       +--> Metadata Preservation
       |
       v
Embedding Model
       |
       v
Persistent ChromaDB


User Query
       |
       v
Retrieval Module
       |
       v
Retrieved Evidence
       |
       v
Generation Module
       |
       v
Grounded Answer + Citations
```

FastAPI will expose the application functionality through an API.

Retrieval and generation will remain independently testable to support systematic evaluation and optimization.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| LLM orchestration | LangChain |
| LLM provider | Groq |
| Initial generation model | `openai/gpt-oss-120b` |
| Embeddings | Sentence Transformers |
| Initial embedding model | `BAAI/bge-small-en-v1.5` |
| Vector database | ChromaDB |
| Document processing | PyPDF |
| API | FastAPI |
| Validation and configuration | Pydantic |
| Testing | Pytest |

## Initial RAG Configuration

These are baseline values intended for experimentation, not universal optimal settings.

| Parameter | Initial Value |
|---|---|
| Embedding dimensions | 384 |
| Chunking strategy | Recursive character splitting |
| Chunk size | 1,000 characters |
| Chunk overlap | 200 characters |
| Retrieval strategy | Similarity search |
| Retrieval top-k | 4 |
| Reranking | Not enabled initially |
| Retrieval score threshold | Not enabled initially |
| Embedding execution | Local CPU |
| Vector storage | Persistent ChromaDB |

These settings will be evaluated and optimized using a retrieval evaluation dataset.

## Document Identity and Metadata

Documents will be identified using SHA-256 content hashes.

- Identical document contents will be detected and skipped.
- Modified documents will receive different identities.
- Document metadata will preserve filename, title, document identity, file hash, and source.
- Chunk metadata will preserve document identity, chunk identity, original PDF page number, and chunk position.
- Page numbers will use the original PDF's 1-based numbering.

## Project Structure

```text
research-knowledge-assistant/
├── app/
│   ├── core/          # Configuration and shared infrastructure
│   ├── ingestion/     # Document loading, processing and indexing
│   ├── retrieval/     # Evidence retrieval
│   ├── generation/    # Prompts and grounded generation
│   ├── schemas/       # Pydantic models
│   └── api/           # FastAPI application and routes
├── data/
│   ├── documents/     # Source research documents
│   └── chroma_db/     # Persistent vector storage
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vaibhav3792/research-knowledge-assistant.git
cd research-knowledge-assistant
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Add your Groq API key to `.env`.

Never commit `.env` or expose API keys publicly.

### 5. Run tests

```powershell
python -m pytest
```

## Configuration

Application settings are centrally managed through Pydantic Settings in `app/core/config.py`.

Environment variables can override defaults.

Important settings include:

- `GROQ_API_KEY`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `EMBEDDING_DEVICE`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_TOP_K`
- `DOCUMENTS_DIR`
- `CHROMA_DIR`

Refer to `.env.example` for the complete configuration template.

## Development Roadmap

1. **Stage 1 — Foundations and Architecture:** Project setup, dependencies, configuration, architecture and design decisions.
2. **Stage 2 — Document Ingestion:** PDF loading, hashing, metadata extraction, splitting and indexing.
3. **Stage 3 — Retrieval Engineering:** Semantic search, retrieval interfaces and independent evaluation.
4. **Stage 4 — Grounded Generation:** Prompt engineering, Groq integration, citations and abstention.
5. **Stage 5 — RAG Integration:** Connect ingestion, retrieval and generation into a complete pipeline.
6. **Stage 6 — Evaluation and Optimization:** Retrieval metrics, answer quality, chunking experiments and retrieval improvements.
7. **Stage 7 — API and Deployment:** FastAPI endpoints, testing, deployment and documentation.

## Project Status

Currently completing Stage 1: Foundations and Architecture.

No document ingestion or RAG application logic has been implemented yet.

## Learning Focus

This project is being developed as a hands-on AI Engineering exercise, focusing on:

- Understanding RAG mechanisms and design trade-offs.
- Building modular and testable LLM applications.
- Evaluating retrieval and generation independently.
- Designing reliable evidence-grounded systems.
- Experimenting with retrieval optimization.
- Deploying a functional application.