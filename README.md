# RAG + Reranker Evaluation Platform

A comprehensive web application for evaluating Retrieval-Augmented Generation (RAG) pipelines with optional reranker models, built on the Open RAGBench dataset.

## Features

- **Dataset Management**: Download and ingest the Vectara Open RAGBench dataset
- **Embedding Support**: HuggingFace and OpenAI embedding models
- **Reranker Integration**: Optional cross-encoder reranking
- **Evaluation Pipeline**: End-to-end RAG evaluation with LLM-as-Judge
- **Results Analysis**: Detailed per-query results and aggregate metrics
- **System Management**: Memory monitoring and model unloading

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-reranker-evaluator.git
   cd rag-reranker-evaluator
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Open your browser to `http://localhost:8501`

### First Evaluation Run

1. **Dataset**: Click "Download & Ingest Dataset" in Section 1
2. **Embedding**: Configure an embedding model in Section 2 (e.g., "sentence-transformers/all-MiniLM-L6-v2")
3. **Evaluation**: Set up evaluation parameters in Section 3, provide an OpenAI API key, and start
4. **Results**: View detailed results in Section 4

## Architecture

- **Backend**: FastAPI with PostgreSQL + pgvector
- **Frontend**: Streamlit web interface
- **Evaluation**: HuggingFace Transformers, OpenAI API, Sentence Transformers

## API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/dataset/ingest` - Ingest dataset
- `POST /api/v1/embedding/models` - Create embedding model
- `POST /api/v1/evaluation/runs` - Start evaluation
- `GET /api/v1/results/{run_id}/details` - Get results

## Development

### Backend

```bash
cd backend
pip install -r ../requirements/backend.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
pip install -r ../requirements/frontend.txt
streamlit run app.py
```

### Database

```bash
docker run --name rag-db -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:15
docker exec -it rag-db psql -U postgres -c "CREATE EXTENSION vector;"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Acknowledgments

- Vectara Open RAGBench dataset
- HuggingFace Transformers
- Sentence Transformers
- OpenAI API
- Streamlit
- FastAPI