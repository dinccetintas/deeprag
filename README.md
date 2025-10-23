# Deep Thinking RAG API

This project converts the experimental "Deep Thinking RAG" notebook into a production-ready, modular Python service. It exposes
a FastAPI REST interface for document ingestion and question answering over a PGVector-backed knowledge base while delegating
reasoning to a multi-step LangGraph agent powered by a local Ollama model.

## Features

- Modular architecture separating API, configuration, and RAG logic.
- Ingestion pipeline that cleans, chunks, embeds, and stores documents in PGVector.
- Hybrid retrieval funnel (vector, BM25 keyword, and RRF fusion) with cross-encoder reranking.
- LangGraph-powered reasoning agent implementing planning, adaptive retrieval, reflection, and synthesis with citations.
- Configurable integrations via environment variables (.env) for Ollama, embeddings, and database connectivity.

## Getting Started

### 1. Environment variables

Copy the provided template and adjust it to match your environment:

```bash
cp .env.example .env
```

Ensure the following values are correct:

- `CONNECTION_STRING`: PGVector/PostgreSQL connection string.
- `EMBEDDING_MODEL_PATH`: Absolute path to the local `instructor-xl` Hugging Face model directory.
- `OLLAMA_BASE_URL`: Base URL of your Ollama server.
- `OLLAMA_MODEL_ID`: Ollama model identifier (e.g., `llama3:70b`).

Additional runtime options (e.g., enabling Tavily web search) can be configured in the `.env` file as described in
`src/core/config.py`.

### 2. Install dependencies

Create a Python 3.10+ virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the API server

Start the FastAPI application with Uvicorn:

```bash
uvicorn src.main:app --reload
```

The server will expose the API at `http://127.0.0.1:8000` by default.

### 4. API usage

#### Health check

```bash
curl http://127.0.0.1:8000/health
```

#### Ingest a document

Provide either a file path or raw content. Metadata is optional and will be attached to every chunk.

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
        "file_path": "./data/nvda_10k_2023_clean.txt",
        "metadata": {"company": "NVIDIA", "filing_year": "2023"}
      }'
```

#### Submit a query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the competitive risks NVIDIA identified and relate them to AMD's 2024 AI strategy."}'
```

Example response:

```json
{
  "answer": "... multi-step analysis with citations ..."
}
```

## Project Structure

```
src/
├── api/
│   └── endpoints.py          # FastAPI routes
├── core/
│   ├── config.py             # Environment-driven settings
│   └── logging_config.py     # Application logging
├── services/
│   ├── rag_pipeline.py       # Deep Thinking RAG pipeline implementation
│   └── vector_store.py       # PGVector helpers
└── main.py                   # FastAPI entry point
```

## Notes

- The service expects a running PostgreSQL/PGVector instance reachable via the configured connection string.
- Ensure the Ollama server is running and has the requested model pulled locally.
- If Tavily search integration is enabled, set a valid `TAVILY_API_KEY` in your environment.

