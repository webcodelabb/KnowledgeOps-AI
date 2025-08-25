"""
Celery application configuration for KnowledgeOps AI
"""
from celery import Celery
from app.config import settings
from app.logging import get_logger

# Configure Celery
celery_app = Celery(
    "knowledgeops_ai",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.document_processing"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
)

# Configure logging
logger = get_logger(__name__)

# Task routing
celery_app.conf.task_routes = {
    "app.tasks.document_processing.*": {"queue": "document_processing"},
}

# Optional: Configure task-specific settings
celery_app.conf.task_annotations = {
    "app.tasks.document_processing.process_document": {
        "rate_limit": "10/m",  # Max 10 tasks per minute
    }
}
