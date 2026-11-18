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

The system follows a modular architecture with separate ingestion, indexing, retrieval, and generation components.

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
Document Chunks
       |
       v
Indexing Module
       |
       +--> Embedding Model
       |
       v
Persistent ChromaDB


User Query
       |
       v
Retrieval Module
       |
       +--> Query Embedding
       +--> Semantic Similarity Search
       +--> Metadata Filtering
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

Retrieval and generation remain independently testable to support systematic evaluation and optimization.

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

These settings can be evaluated and optimized using the retrieval evaluation dataset.

## Document Identity and Metadata

Documents are identified using SHA-256 hashes of their original file contents.

- Identical file contents receive the same document identity.
- Modified file contents receive a different identity.
- Duplicate detection currently uses an in-memory registry.
- Document metadata preserves the source filename and source path.
- PDF page metadata preserves the original 1-based page number.
- The ingestion pipeline attaches the stable `document_id` before chunking.

### Document Chunking

Documents are split using LangChain's `RecursiveCharacterTextSplitter`.

- PDF pages are split independently to preserve page-level provenance.
- TXT documents are split as a single source document.
- Empty and whitespace-only documents are ignored during chunking.
- Chunk size and overlap are configurable through application settings.
- Each chunk retains its original source metadata.

Initial chunking configuration:

| Parameter | Value |
|---|---|
| Chunk size | 1,000 characters |
| Chunk overlap | 200 characters |
| Splitting strategy | Recursive character splitting |
| Start index tracking | Enabled |

### Chunk Metadata

Each generated chunk preserves its original document metadata and adds:

- `chunk_id`: Deterministic SHA-256 identifier for the chunk.
- `document_id`: Identity of the original file.
- `page_number`: Original 1-based PDF page number, when applicable.
- `chunk_index`: Zero-based position of the chunk within its source document or PDF page.
- `start_index`: Character position of the chunk within its original source content.

Chunk identities are generated using the document identity, source segment, chunk index, and chunk content.

The ingestion pipeline registers a document only after loading and chunking complete successfully.

The document registry currently performs in-memory duplicate detection during ingestion.

Persistent vector indexing is implemented separately through ChromaDB. Deterministic chunk IDs are used as vector database record IDs, allowing repeated indexing of the same chunks to update existing records instead of creating duplicates.

The ingestion pipeline and vector indexing module remain separate.

## Embeddings and Vector Indexing

The project uses Sentence Transformers to generate local document embeddings.

Initial embedding configuration:

| Parameter | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimensions | 384 |
| Execution device | CPU |
| Embedding normalization | Enabled |
| Vector database | ChromaDB |
| Collection name | `research_documents` |
| Persistence | Local persistent storage |

The embedding model is initialized through a cached shared instance to avoid repeatedly loading model weights.

ChromaDB is configured with the shared embedding model and a persistent storage directory.

### Indexing

The indexing module accepts chunk-level LangChain Documents and:

- Validates that every chunk has a valid `chunk_id`.
- Rejects duplicate chunk IDs within an indexing batch.
- Uses deterministic chunk IDs as ChromaDB record IDs.
- Stores chunk content and metadata.
- Supports updating existing records when the same chunk IDs are indexed again.
- Propagates vector database errors to the caller.

Integration tests verify storage, re-indexing behavior, and persistence after reopening the database.

The vector database files are stored locally and excluded from Git.

## Retrieval Engineering

The retrieval module performs semantic similarity search over the persistent ChromaDB collection.

### Semantic Retrieval

The retrieval implementation:

- Converts user queries into embeddings using the configured Sentence Transformers model.
- Retrieves relevant document chunks using ChromaDB similarity search.
- Supports configurable `top_k`, with the application retrieval setting used by default.
- Preserves chunk content and source metadata in retrieved LangChain Documents.
- Uses the production vector store without coupling retrieval to document ingestion.
- Propagates vector database errors to the caller.

### Metadata Filtering

Retrieval supports filtering results using the following metadata fields:

| Field | Purpose |
|---|---|
| `document_id` | Restrict retrieval to a specific source document |
| `filename` | Restrict retrieval to a specific filename |
| `page_number` | Restrict retrieval to a specific PDF page |

Multiple supported filters can be combined using logical AND semantics.

Unsupported filter fields and invalid filter values are rejected through input validation.

### Retrieval Usage Example

```python
from app.retrieval.retriever import retrieve_documents

results = retrieve_documents(
    query="What is retrieval-augmented generation?",
    top_k=4,
    filters={"filename": "research_paper.pdf"},
)

for document in results:
    print(document.page_content)
    print(document.metadata)
```

The `top_k` parameter controls the maximum number of chunks returned. When omitted, the configured default retrieval value is used.

## Retrieval Evaluation

Retrieval quality is evaluated independently of LLM generation using a manually curated evaluation dataset.

The initial dataset contains six synthetic document chunks and six evaluation queries with manually assigned relevant chunk IDs.

The evaluation uses:

- The actual configured BGE embedding model.
- An isolated in-memory ChromaDB collection.
- The retrieval evaluation metrics implemented in `app/retrieval/evaluation.py`.

The evaluation process does not modify the production vector database.

### Evaluation Metrics

| Metric | Definition |
|---|---|
| Precision@K | Number of relevant retrieved chunks divided by K |
| Recall@K | Number of relevant retrieved chunks divided by the total number of known relevant chunks |
| Hit Rate@K | 1 if at least one relevant chunk is retrieved; otherwise 0 |

Metrics are calculated independently for each query and then macro-averaged across all evaluation queries.

### Running the Evaluation

Run the evaluation using the default retrieval depth:

```powershell
python -m scripts.evaluate_retrieval
```

Specify a different retrieval depth:

```powershell
python -m scripts.evaluate_retrieval --top-k 2
```

The evaluation dataset is stored at:

```text
data/evaluation/retrieval_eval.json
```

The evaluation script validates the dataset before running retrieval. Validation includes:

- Dataset structure and required fields.
- Unique chunk and query IDs.
- Non-empty document content and query text.
- Valid metadata structures and scalar values.
- Finite numeric metadata values.
- Valid relevant chunk references.
- Rejection of duplicate or unknown relevant chunk IDs.

### Initial Retrieval Baseline

The following results were obtained using the actual BGE embedding model and the six-query evaluation dataset.

| Metric | K = 2 | K = 4 |
|---|---:|---:|
| Precision@K | 0.500 | 0.292 |
| Recall@K | 0.750 | 0.833 |
| Hit Rate@K | 1.000 | 1.000 |

### Baseline Analysis

The results illustrate the trade-off between retrieval precision and recall.

- Increasing K from 2 to 4 improves recall from 0.750 to 0.833.
- Precision decreases from 0.500 to 0.292 as more chunks are retrieved.
- Both configurations achieve a hit rate of 1.000 on the current dataset.

A smaller K returns fewer chunks and can reduce irrelevant context, while a larger K can improve evidence coverage at the cost of introducing additional context.

**Evaluation limitation:** The current benchmark contains only six synthetic chunks and six manually curated queries. These results are an initial implementation baseline, not evidence of production retrieval quality. A larger and more representative evaluation dataset is required before drawing meaningful performance conclusions.

## Project Structure

```text
research-knowledge-assistant/
|
+-- app/
|   |
|   +-- core/
|   |   +-- config.py
|   |   +-- embeddings.py
|   |   +-- vector_store.py
|   |
|   +-- ingestion/
|   |   +-- chunker.py
|   |   +-- document_identity.py
|   |   +-- document_registry.py
|   |   +-- indexer.py
|   |   +-- loader.py
|   |   +-- pdf_loader.py
|   |   +-- pipeline.py
|   |   +-- txt_loader.py
|   |
|   +-- retrieval/
|       +-- __init__.py
|       +-- retriever.py
|       +-- evaluation.py
|
+-- data/
|   +-- raw/
|   +-- documents/
|   +-- chroma_db/
|   +-- evaluation/
|       +-- retrieval_eval.json
|
+-- scripts/
|   +-- evaluate_retrieval.py
|
+-- tests/
|   +-- unit/
|   |   +-- test_chunker.py
|   |   +-- test_document_identity.py
|   |   +-- test_document_registry.py
|   |   +-- test_indexer.py
|   |   +-- test_ingestion_pipeline.py
|   |   +-- test_loader.py
|   |   +-- test_pdf_loader.py
|   |   +-- test_txt_loader.py
|   |   +-- test_retriever.py
|   |   +-- test_retrieval_evaluation.py
|   |   +-- test_evaluate_retrieval.py
|   |
|   +-- integration/
|       +-- test_vector_store.py
|       +-- test_retriever_integration.py
|
+-- .env.example
+-- .gitignore
+-- README.md
+-- requirements.txt
```

Raw research PDFs and generated vector database files are excluded from Git.

Generation, schemas, and API modules will be added in their respective development stages.

## Setup

### 1. Clone the repository

```powershell
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

### 6. Run retrieval evaluation

```powershell
python -m scripts.evaluate_retrieval
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
2. **Stage 2 — Document Ingestion:** PDF and TXT loading, document hashing, duplicate detection, metadata handling, and ingestion pipeline. Completed.
3. **Stage 3 — Document Processing and Chunking:** Recursive character splitting, metadata propagation, deterministic chunk identities, pipeline integration, and chunking tests. Completed.
4. **Stage 4 — Embeddings and Vector Indexing:** Local embedding integration, persistent ChromaDB, deterministic chunk indexing, duplicate-safe upserts, and persistence tests. Completed.
5. **Stage 5 — Retrieval Engineering:** Semantic similarity search, metadata filtering, source tracking, retrieval metrics, evaluation dataset, and robustness testing. Implementation and testing completed; final documentation and Git review in progress.
6. **Stage 6 — RAG Generation:** Groq integration, prompt construction, context assembly, grounded answers, citations, and abstention.
7. **Stage 7 — FastAPI Application:** Request and response schemas, endpoints, validation, error handling, and API tests.
8. **Stage 8 — Evaluation and Optimization:** Retrieval metrics, answer quality, failure analysis, latency, and reliability improvements.
9. **Stage 9 — Deployment:** Production configuration, deployment, environment secrets, persistent storage, and live verification.

## Project Status

**Current stage: Stage 5 — Retrieval Engineering**

### Implemented

- PDF loading with page-level extraction.
- TXT loading with UTF-8 validation.
- File-type-based document loader.
- SHA-256 document identity.
- In-memory duplicate registry.
- Ingestion pipeline with document identity metadata.
- Recursive character-based document chunking.
- PDF page-level provenance preservation.
- Chunk metadata propagation, including start index tracking.
- Deterministic SHA-256 chunk identities.
- Integration of chunking into the ingestion pipeline.
- Registration only after successful loading and chunking.
- Cached Sentence Transformers embedding model initialization.
- Local BGE embedding generation with 384-dimensional normalized embeddings.
- Persistent ChromaDB vector store initialization.
- Chunk indexing using deterministic IDs.
- Validation of missing and duplicate chunk IDs.
- Duplicate-safe re-indexing through ChromaDB upserts.
- ChromaDB persistence and reopening verification.
- Semantic retrieval using ChromaDB similarity search.
- Configurable retrieval depth.
- Metadata filtering by document ID, filename, and page number.
- Combined metadata filtering.
- Retrieval input validation and error handling.
- Precision@K, Recall@K, and Hit Rate@K implementation.
- Evaluation dataset and dataset validation.
- Retrieval evaluation using real BGE embeddings.
- K=2 and K=4 baseline comparisons.
- Unit and integration tests for ingestion, chunking, indexing, vector storage, retrieval, and evaluation.

**Current test suite: 172 passing tests.**

Stage 5 implementation and testing are complete. Final documentation review, Git commit, and push remain.

RAG generation, API development, expanded evaluation, optimization, and deployment are the remaining stages.

## Learning Focus

This project emphasizes practical AI Engineering skills:

- Understanding RAG mechanisms and design trade-offs.
- Building modular and testable LLM applications.
- Designing reliable document ingestion pipelines.
- Implementing semantic retrieval and metadata filtering.
- Evaluating retrieval and generation independently.
- Experimenting with retrieval optimization.
- Building evidence-grounded systems.
- Deploying a functional AI application.
