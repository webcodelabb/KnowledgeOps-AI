#  KnowledgeOps AI

End-to-end document intelligence platform: ingest → embed → retrieve → answer → monitor.

## Screenshots

<div align="center">
  <img src="docs/img1.png" alt="Main Dashboard" width="45%" />
  <img src="docs/img2.png" alt="Query Interface" width="45%" />
  <br>
  <img src="docs/img3.png" alt="Document Processing" width="45%" />
  <img src="docs/img4.png" alt="Analytics Dashboard" width="45%" />
  <br>
  <img src="docs/img5.png" alt="Grafana Monitoring" width="45%" />
  <img src="docs/img6.png" alt="Evaluation Report" width="45%" />
</div>

### Screenshot Descriptions

| Screenshot | Description |
|------------|-------------|
| **Main Dashboard** | Overview of the KnowledgeOps AI platform with document uploads, query performance metrics, and system health indicators |
| **Query Interface** | AI-powered question-answering interface with confidence scores, source attribution, and chat-like experience |
| **Document Processing** | File upload and processing workflow with real-time status updates and progress tracking |
| **Analytics Dashboard** | Performance analytics with query volume, response times, success rates, and data visualizations |
| **Grafana Monitoring** | Real-time system monitoring with Prometheus metrics, query performance, and token usage tracking |
| **Evaluation Report** | RAG pipeline evaluation results with relevance@k scores, faithfulness metrics, and performance analysis |

## Overview

KnowledgeOps AI is a comprehensive document intelligence platform that provides:

- **Document Ingestion**: Support for PDFs, Docs, HTML, and URLs
- **Vector Search**: Hybrid BM25 + vector retrieval
- **RAG Pipeline**: Context-aware question answering
- **Observability**: Structured logging and Prometheus metrics
- **Multi-tenant**: JWT authentication and RBAC

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Celery/RQ     │    │   PostgreSQL    │
│   (REST API)    │───▶│   (Async Jobs)  │───▶│   + pgvector    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │     Redis       │    │   Vector DB     │
│   (Metrics)     │    │   (Queue/Cache) │    │   (Embeddings)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Implemented
- FastAPI application with structured logging
- Pydantic v2 models for request/response validation
- Async SQLAlchemy database integration
- Prometheus metrics collection
- Health check and monitoring endpoints
- CORS middleware
- Global exception handling
- Environment-based configuration

### 🚧 In Progress
- Document ingestion pipeline
- RAG query processing
- Authentication and authorization
- Admin UI

### ✅ Database Setup Complete
- PostgreSQL with pgvector extension
- Alembic migrations for schema management
- SQLAlchemy models for documents, chunks, conversations
- Vector similarity search support
- Multi-tenant organization support

### ✅ Document Processing Complete
- Celery worker for async document processing
- LangChain document loaders (PDF, HTML, Text)
- Text chunking with configurable size and overlap
- OpenAI embeddings generation
- Structured logging and progress tracking
- Batch document processing support

### ✅ Advanced Retrieval Complete
- Vector similarity search with pgvector
- BM25 reranking for improved relevance
- Chunk deduplication to avoid redundancy
- Token budget enforcement for LLM context
- RetrievalQA chain with confidence scoring
- Source attribution and metadata tracking

### ✅ Intelligent QA Agent Complete
- LangGraph-based query reformulation agent
- Automatic confidence threshold checking
- Keyword extraction from top chunks
- Metadata filter extraction and application
- Multi-attempt comparison and selection
- Detailed observability with attempt scores

### ✅ Prometheus Metrics Complete
- Comprehensive query performance metrics
- Token usage tracking (input/output)
- Retrieval performance monitoring
- Intelligent agent metrics
- Grafana dashboard with 12 panels
- Real-time observability and alerting

### ✅ RAG Evaluation System Complete
- Comprehensive evaluation pipeline with CSV input
- Relevance@k scoring (k=1,3,5,10) for retrieval quality
- Faithfulness scoring using LLM-as-judge
- HTML reports with interactive charts
- Processing time and error tracking
- Sample evaluation dataset included

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- Redis (for Celery)

## Evaluation

### RAG Pipeline Evaluation
The platform includes a comprehensive evaluation system for testing RAG pipeline performance:

```bash
# Test evaluation system
python test_eval.py

# Run evaluation on sample data
python scripts/eval.py scripts/sample_eval_data.csv --openai-key YOUR_KEY

# Run evaluation with custom parameters
python scripts/eval.py your_eval_data.csv \
  --output custom_report.html \
  --api-url http://localhost:8000 \
  --org-id your-org \
  --max-examples 50
```

### Evaluation Metrics
- **Relevance@k**: Measures retrieval quality at different k values (1, 3, 5, 10)
- **Faithfulness**: LLM-as-judge scoring for answer faithfulness to source material
- **Processing Time**: Performance tracking for optimization
- **Error Tracking**: Comprehensive error reporting and analysis

### CSV Format
The evaluation script expects a CSV with the following columns:
- `question`: The query to evaluate
- `gold_doc_id`: Expected document ID (optional)
- `gold_text`: Expected text content (optional)

### HTML Reports
Generated reports include:
- Performance summary with key metrics
- Interactive charts for relevance and faithfulness distributions
- Detailed results for each evaluation example
- Error analysis for failed evaluations
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd knowledgeops-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Set up database**
   ```bash
   # Create PostgreSQL database with pgvector extension
   createdb knowledgeops
   psql knowledgeops -c "CREATE EXTENSION IF NOT EXISTS vector;"
   
   # Run database migrations
   python migrate.py
   ```

5. **Start services**
   ```bash
   # Start Redis (for Celery)
   redis-server
   
   # Start Celery worker (in new terminal)
   python worker.py
   
   # Start FastAPI application
   python run.py
   ```

5. **Run the application**
   ```bash
   python -m app.main
   ```

The API will be available at `http://localhost:8000`

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## API Endpoints

### Health & Monitoring
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation

### Document Management
- `POST /ingest` - Ingest documents for processing
- `POST /ingest/batch` - Ingest multiple documents in batch
- `GET /documents` - List ingested documents
- `GET /documents/{id}` - Get document details
- `GET /stats/{org_id}` - Get organization statistics
- `GET /tasks/{task_id}` - Get Celery task status

### Query Interface
- `POST /query` - Query documents using RAG with answer generation
- `POST /retrieve` - Retrieve relevant documents without answer generation
- `POST /query/intelligent` - Intelligent QA with query reformulation
- `GET /queries` - List recent queries
- `GET /queries/{id}` - Get query details

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://user:password@localhost/knowledgeops` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `OPENAI_API_KEY` | OpenAI API key | `None` |
| `SECRET_KEY` | JWT secret key | `your-secret-key-change-in-production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEBUG` | Debug mode | `false` |

## Development

### Project Structure

```
knowledgeops-ai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── database.py          # Database setup
│   ├── models.py            # Pydantic models
│   ├── models_db.py         # SQLAlchemy database models
│   ├── db_utils.py          # Database utility functions
│   ├── logging.py           # Structured logging
│   ├── metrics.py           # Prometheus metrics
│   ├── retrieval.py         # Advanced retrieval system
│   ├── agent.py             # LangGraph intelligent agent
│   ├── celery_app.py        # Celery configuration
│   └── tasks/
│       ├── __init__.py      # Tasks package
│       └── document_processing.py  # Document processing tasks
├── alembic/                 # Database migrations
│   ├── env.py              # Alembic environment
│   ├── script.py.mako      # Migration template
│   └── versions/           # Migration files
├── tests/                   # Test suite
├── docker/                  # Docker configuration
├── requirements.txt         # Python dependencies
├── env.example             # Environment template
├── run.py                  # Startup script
├── worker.py               # Celery worker script
├── demo.py                 # API demo
├── migrate.py              # Database migration script
├── test_setup.py           # Setup verification
├── test_db.py              # Database setup verification
├── test_celery.py          # Celery setup verification
├── test_retrieval.py       # Retrieval system verification
├── test_agent.py           # LangGraph agent verification
├── test_metrics.py         # Prometheus metrics verification
├── test_eval.py            # RAG evaluation system verification
├── grafana-dashboard.json  # Grafana dashboard configuration
├── scripts/                # Evaluation scripts
│   ├── eval.py             # RAG evaluation pipeline
│   └── sample_eval_data.csv # Sample evaluation dataset
├── DATABASE.md             # Database documentation
└── README.md               # This file
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

## Monitoring

### Metrics

The application exposes Prometheus metrics at `/metrics`:

- HTTP request counts and durations
- Document ingestion job status
- Query processing metrics
- Database connection counts
- System resource usage

### Logging

Structured JSON logging is configured with:

- Request/response logging
- Error tracking with context
- Performance metrics
- Business event logging

### Health Checks

Health check endpoint at `/health` provides:

- Application status
- Database connectivity
- External service status
- Version information

## Deployment

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Configure proper `SECRET_KEY`
- [ ] Set up PostgreSQL with pgvector
- [ ] Configure Redis for Celery
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure reverse proxy (nginx)
- [ ] Set up SSL certificates
- [ ] Configure backup strategy

### Docker Deployment

```bash
# Build production image
docker build -t knowledgeops-ai .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  knowledgeops-ai
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API documentation at `/redoc`
