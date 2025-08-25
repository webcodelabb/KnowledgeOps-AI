"""
Prometheus metrics configuration
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Business metrics
INGESTION_JOBS = Counter(
    'document_ingestion_jobs_total',
    'Total document ingestion jobs',
    ['status']
)

QUERY_REQUESTS = Counter(
    'query_requests_total',
    'Total query requests',
    ['status']
)

QUERY_DURATION = Histogram(
    'query_duration_seconds',
    'Query processing duration in seconds'
)

# Detailed query metrics
QUERY_TOKENS_IN = Counter(
    'query_tokens_in_total',
    'Total input tokens processed',
    ['method']
)

QUERY_TOKENS_OUT = Counter(
    'query_tokens_out_total',
    'Total output tokens generated',
    ['method']
)

QUERY_RETRIEVED_CHUNKS = Histogram(
    'query_retrieved_chunks',
    'Number of chunks retrieved per query',
    ['method'],
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
)

QUERY_ANSWER_CONFIDENCE = Histogram(
    'query_answer_confidence',
    'Answer confidence scores',
    ['method'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

QUERY_PROCESSING_TIME = Histogram(
    'query_processing_time_seconds',
    'Query processing time in seconds',
    ['method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Intelligent agent metrics
INTELLIGENT_QUERY_ATTEMPTS = Histogram(
    'intelligent_query_attempts',
    'Number of attempts per intelligent query',
    ['method'],
    buckets=[1, 2, 3, 4, 5]
)

INTELLIGENT_QUERY_REFORMULATIONS = Counter(
    'intelligent_query_reformulations_total',
    'Total number of query reformulations',
    ['method']
)

# Retrieval metrics
RETRIEVAL_REQUESTS = Counter(
    'retrieval_requests_total',
    'Total number of retrieval requests',
    ['method', 'status']
)

RETRIEVAL_LATENCY = Histogram(
    'retrieval_latency_seconds',
    'Retrieval request latency in seconds',
    ['method'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'database_active_connections',
    'Number of active database connections'
)

DOCUMENT_COUNT = Gauge(
    'documents_total',
    'Total number of documents in the system'
)

EMBEDDING_COUNT = Gauge(
    'embeddings_total',
    'Total number of embeddings in the vector database'
)


def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest()


def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def record_ingestion_job(status: str):
    """Record document ingestion job"""
    INGESTION_JOBS.labels(status=status).inc()


def record_query_request(status: str, duration: float):
    """Record query request"""
    QUERY_REQUESTS.labels(status=status).inc()
    QUERY_DURATION.observe(duration)


def record_detailed_query_metrics(
    method: str,
    tokens_in: int,
    tokens_out: int,
    retrieved_chunks: int,
    confidence: float,
    processing_time: float
):
    """Record detailed query metrics"""
    QUERY_TOKENS_IN.labels(method=method).inc(tokens_in)
    QUERY_TOKENS_OUT.labels(method=method).inc(tokens_out)
    QUERY_RETRIEVED_CHUNKS.labels(method=method).observe(retrieved_chunks)
    QUERY_ANSWER_CONFIDENCE.labels(method=method).observe(confidence)
    QUERY_PROCESSING_TIME.labels(method=method).observe(processing_time)


def record_intelligent_query_metrics(
    method: str,
    attempts: int,
    reformulations: int
):
    """Record intelligent query metrics"""
    INTELLIGENT_QUERY_ATTEMPTS.labels(method=method).observe(attempts)
    if reformulations > 0:
        INTELLIGENT_QUERY_REFORMULATIONS.labels(method=method).inc(reformulations)


def record_retrieval_metrics(
    method: str,
    status: str,
    duration: float
):
    """Record retrieval metrics"""
    RETRIEVAL_REQUESTS.labels(method=method, status=status).inc()
    RETRIEVAL_LATENCY.labels(method=method).observe(duration)


def update_document_count(count: int):
    """Update document count metric"""
    DOCUMENT_COUNT.set(count)


def update_embedding_count(count: int):
    """Update embedding count metric"""
    EMBEDDING_COUNT.set(count)


def update_active_connections(count: int):
    """Update active connections metric"""
    ACTIVE_CONNECTIONS.set(count)
