# RAG + Reranker Evaluation Platform

Purpose-built to benchmark RAG pipelines on Open RAGBench with optional rerankers and an LLM-as-judge scoring loop.

## How the Judge Works
- For each query, the system retrieves contexts, optionally reranks them, generates an answer, and passes everything to a judge LLM.
- The judge scores answer quality (groundedness, completeness, usefulness) and stores per-query details plus aggregate metrics.
- Because the judge runs on every turn, results stay comparable across embedding/reranker/model choices.

## Essential Flow
1) Ingest: Download and load Open RAGBench into Postgres/pgvector.
2) Embed: Create an embedding model (HF or OpenAI) and vectorize the corpus.
3) Retrieve (optionally rerank): Fetch top contexts; rerank with a cross-encoder if enabled.
4) Generate: Call the chosen LLM to draft an answer.
5) Judge: Score each answer with the judge LLM and record metrics.

## Quick Start
- Prereqs: Docker + Docker Compose, Git.
- Run:
  ```bash
  git clone https://github.com/yourusername/rag-reranker-evaluator.git
  cd rag-reranker-evaluator
  docker-compose up -d
  ```
- Open `http://localhost:8501` and follow the sections: ingest dataset, configure embedding, set evaluation params (API keys), start run, review results.

## Stack
- Backend: FastAPI + Postgres/pgvector
- Frontend: Streamlit
- Models: HuggingFace / OpenAI embeddings, optional cross-encoder reranker, LLM-as-judge

## Key Endpoints
- `POST /api/v1/dataset/ingest` — ingest dataset
- `POST /api/v1/embedding/models` — create embedding model
- `POST /api/v1/evaluation/runs` — start evaluation
- `GET /api/v1/results/{run_id}/details` — per-query results

## Dev Tips
- Backend: `cd backend && pip install -r ../requirements/backend.txt && uvicorn main:app --reload`
- Frontend: `cd frontend && pip install -r ../requirements/frontend.txt && streamlit run app.py`
- DB (local alt):
  ```bash
  docker run --name rag-db -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:15
  docker exec -it rag-db psql -U postgres -c "CREATE EXTENSION vector;"
  ```
