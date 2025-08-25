"""
Pydantic models for request/response schemas
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "0.1.0"


class IngestRequest(BaseModel):
    """Document ingestion request"""
    file_url: Optional[HttpUrl] = Field(None, description="URL to document")
    file_path: Optional[str] = Field(None, description="Local file path")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    chunk_size: int = Field(default=800, ge=100, le=2000, description="Token chunk size")
    overlap: int = Field(default=100, ge=0, le=500, description="Chunk overlap")


class IngestResponse(BaseModel):
    """Document ingestion response"""
    job_id: str = Field(description="Async job ID")
    status: str = Field(description="Job status")
    message: str = Field(description="Status message")
    document_id: Optional[str] = Field(None, description="Document ID")


class QueryRequest(BaseModel):
    """Query request"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Query filters")
    include_sources: bool = Field(default=True, description="Include source documents")


class QueryResponse(BaseModel):
    """Query response"""
    answer: str = Field(description="Generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(ge=0.0, le=1.0, description="Answer confidence")
    query_id: str = Field(description="Query ID for tracking")
    processing_time: float = Field(description="Processing time in seconds")


class DocumentMetadata(BaseModel):
    """Document metadata"""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    status: str
    chunk_count: int
    metadata: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
