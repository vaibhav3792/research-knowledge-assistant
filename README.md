# Research & Knowledge Assistant

An AI-powered document intelligence system built using **LangChain and Retrieval-Augmented Generation (RAG)**.

The application is designed to ingest research documents, retrieve relevant evidence, and generate grounded answers with document and page-level citations. Planned capabilities also include research summarization, comparison, and evidence-based abstention.

## Project Objectives

- Ingest and process multiple research documents.
- Retrieve relevant passages using semantic search.
- Generate answers grounded in retrieved evidence.
- Cite supporting documents and original PDF pages.
- Summarize and compare research findings.
- Abstain when retrieved evidence is insufficient.
- Evaluate and optimize retrieval and generation independently.
- Deploy a functional, accessible application.

## Architecture

The system follows a modular architecture with separate ingestion, retrieval, and generation components.

```text
Research Documents
       |
       v
Ingestion Pipeline
       |
       +--> PDF / TXT Loading
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

FastAPI will expose application functionality through an API.

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
| Document processing | pypdf |
| API | FastAPI |
| Configuration | Pydantic Settings |
| Testing | pytest |

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

Documents are identified using SHA-256 hashes of their original file contents.

- Identical file contents receive the same document identity.
- Modified file contents receive a different identity.
- Duplicate detection currently uses an in-memory registry.
- Document metadata preserves the source filename and source path.
- PDF page metadata preserves the original 1-based page number.
- The ingestion pipeline attaches the stable `document_id` to each loaded page.

The planned chunk-level metadata includes:

- `chunk_id`
- `document_id`
- `page_number`
- `chunk_index`

Persistent duplicate detection and indexing will be implemented alongside ChromaDB integration.

## Project Structure

```text
research-knowledge-assistant/
├── app/
│   ├── core/          # Configuration and shared infrastructure
│   ├── ingestion/     # Document loading and ingestion
│   ├── retrieval/     # Evidence retrieval
│   ├── generation/    # Prompts and grounded generation
│   ├── schemas/       # Pydantic models
│   └── api/           # FastAPI application and routes
├── data/
│   ├── raw/           # Local source research documents
│   ├── documents/     # Processed document storage
│   └── chroma_db/     # Persistent vector storage
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Raw research PDFs and generated vector database files are excluded from Git.

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

```powershell
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
python -m pytest -v
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

1. **Stage 1 — Foundations and Architecture:** Project setup, dependencies, configuration, architecture, and design decisions. Completed.
2. **Stage 2 — Document Ingestion:** PDF and TXT loading, document hashing, duplicate detection, metadata handling, and ingestion pipeline. Implementation completed; final verification and commit pending.
3. **Stage 3 — Document Processing and Chunking:** Recursive character splitting, metadata propagation, stable chunk identities, and chunking tests.
4. **Stage 4 — Embeddings and Vector Indexing:** Embedding integration, persistent ChromaDB, indexing, and duplicate-safe persistence.
5. **Stage 5 — Retrieval Engineering:** Similarity search, metadata filtering, source tracking, and retrieval evaluation.
6. **Stage 6 — RAG Generation:** Groq integration, prompt construction, context assembly, grounded answers, citations, and abstention.
7. **Stage 7 — FastAPI Application:** Request and response schemas, endpoints, validation, error handling, and API tests.
8. **Stage 8 — Evaluation and Optimization:** Retrieval metrics, answer quality, failure analysis, latency, and reliability improvements.
9. **Stage 9 — Deployment:** Production configuration, deployment, environment secrets, persistent storage, and live verification.

## Project Status

**Current stage: Stage 2 — Document Ingestion**

Implemented:

- PDF loading with page-level extraction.
- TXT loading with UTF-8 validation.
- File-type-based document loader.
- SHA-256 document identity.
- In-memory duplicate registry.
- Ingestion pipeline with document identity metadata.
- Cached Sentence Transformers embedding model initialization.
- Unit tests covering loaders, document identity, duplicate detection, and ingestion behavior.

The current test suite has 38 passing tests.

Chunking, persistent vector indexing, retrieval, generation, API development, and deployment remain future stages.

## Learning Focus

This project emphasizes practical AI Engineering skills:

- Understanding RAG mechanisms and design trade-offs.
- Building modular and testable LLM applications.
- Designing reliable document ingestion pipelines.
- Evaluating retrieval and generation independently.
- Experimenting with retrieval optimization.
- Building evidence-grounded systems.
- Deploying a functional AI application.
