"""
Main FastAPI application for KnowledgeOps AI
"""
import time
import uuid
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db, init_db, close_db
from app.logging import configure_logging, get_logger, log_request, log_error
from app.metrics import (
    get_metrics, record_request, record_detailed_query_metrics, 
    record_intelligent_query_metrics, record_retrieval_metrics
)
from app.models import (
    HealthResponse, IngestRequest, IngestResponse, 
    QueryRequest, QueryResponse, ErrorResponse
)
from app.celery_app import celery_app
from app.models_db import Document, Chunk, Conversation
from app.db_utils import (
    create_document, get_document_by_id, get_documents_by_org,
    create_chunk, get_chunks_by_document, get_document_stats
)
from app.tasks.document_processing import process_document, process_document_batch
from app.retrieval import RetrievalQAChain, AdvancedRetriever
from app.agent import IntelligentQAAgent
from langchain.embeddings.openai import OpenAIEmbeddings


# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting KnowledgeOps AI application")
    await init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down KnowledgeOps AI application")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="End-to-end document intelligence platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Middleware for request logging and metrics"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log request
    log_request(
        logger,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    # Record metrics
    record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration
    )
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    log_error(logger, exc, {"path": request.url.path, "method": request.method})
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else "An unexpected error occurred"
        ).model_dump()
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse()


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import CONTENT_TYPE_LATEST
    return Response(
        content=get_metrics(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db)
):
    """Ingest a document for processing"""
    try:
        # Create document record
        org_id = request.metadata.get("org_id", "default") if request.metadata else "default"
        author = request.metadata.get("author") if request.metadata else None
        
        document = await create_document(
            session=db,
            org_id=org_id,
            source=str(request.file_url) if request.file_url else request.file_path or "",
            author=author
        )
        
        # Queue document processing task
        task = process_document.delay(
            document_id=str(document.id),
            file_url=str(request.file_url) if request.file_url else None,
            file_path=request.file_path,
            org_id=org_id,
            chunk_size=request.chunk_size,
            chunk_overlap=request.overlap,
            metadata=request.metadata
        )
        
        logger.info("Document processing task queued", 
                   task_id=task.id,
                   document_id=str(document.id),
                   org_id=org_id)
        
        return IngestResponse(
            job_id=task.id,
            status="queued",
            message="Document processing task queued successfully",
            document_id=str(document.id)
        )
        
    except Exception as e:
        log_error(logger, e, {"document_id": str(document.id) if 'document' in locals() else None})
        raise HTTPException(status_code=500, detail="Failed to ingest document")


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Query documents using RAG"""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Get organization ID from request or use default
        org_id = request.filters.get("org_id", "default") if request.filters else "default"
        
        # Initialize embeddings model
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(request.query)
        
        # Initialize QA chain
        qa_chain = RetrievalQAChain(
            session=db,
            org_id=org_id,
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            max_tokens=4000
        )
        
        # Get answer using RAG
        qa_result = await qa_chain.answer_question(
            query=request.query,
            query_embedding=query_embedding,
            top_k=request.top_k
        )
        
        # Format sources for response
        sources = []
        for source in qa_result.sources:
            sources.append({
                "document_id": source["document_id"],
                "chunk_id": source["chunk_id"],
                "title": source["title"],
                "url": source["url"],
                "score": source["score"],
                "content": source["text_preview"]
            })
        
        processing_time = time.time() - start_time
        
        # Record detailed query metrics
        record_detailed_query_metrics(
            method="query",
            tokens_in=qa_result.total_tokens,
            tokens_out=len(qa_result.answer.split()),  # Approximate output tokens
            retrieved_chunks=len(sources),
            confidence=qa_result.confidence,
            processing_time=processing_time
        )
        
        logger.info("Query processed successfully", 
                   query_id=query_id,
                   query=request.query[:100],
                   answer_length=len(qa_result.answer),
                   confidence=qa_result.confidence,
                   sources_count=len(sources),
                   processing_time=processing_time)
        
        return QueryResponse(
            answer=qa_result.answer,
            sources=sources,
            confidence=qa_result.confidence,
            query_id=query_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_error(logger, e, {"query_id": query_id, "processing_time": processing_time})
        raise HTTPException(status_code=500, detail="Failed to process query")


@app.get("/documents", tags=["Documents"])
async def list_documents(
    org_id: str = "default",
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List documents for an organization"""
    try:
        documents = await get_documents_by_org(db, org_id, limit, offset)
        return {
            "documents": [
                {
                    "id": str(doc.id),
                    "org_id": doc.org_id,
                    "source": doc.source,
                    "author": doc.author,
                    "created_at": doc.created_at.isoformat(),
                    "chunk_count": len(doc.chunks)
                }
                for doc in documents
            ],
            "total": len(documents)
        }
    except Exception as e:
        log_error(logger, e, {"org_id": org_id})
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.get("/documents/{document_id}", tags=["Documents"])
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get document details by ID"""
    try:
        doc_uuid = uuid.UUID(document_id)
        document = await get_document_by_id(db, doc_uuid)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": str(document.id),
            "org_id": document.org_id,
            "source": document.source,
            "author": document.author,
            "created_at": document.created_at.isoformat(),
            "chunks": [
                {
                    "id": str(chunk.id),
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "metadata": chunk.metadata,
                    "has_embedding": chunk.embedding is not None
                }
                for chunk in document.chunks
            ]
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        log_error(logger, e, {"document_id": document_id})
        raise HTTPException(status_code=500, detail="Failed to get document")


@app.get("/stats/{org_id}", tags=["Statistics"])
async def get_organization_stats(
    org_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get statistics for an organization"""
    try:
        stats = await get_document_stats(db, org_id)
        return {
            "org_id": org_id,
            "statistics": stats
        }
    except Exception as e:
        log_error(logger, e, {"org_id": org_id})
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task_status(task_id: str):
    """Get Celery task status and result"""
    try:
        task = celery_app.AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "status": task.status,
            "result": task.result if task.ready() else None
        }
        
        if task.status == "PROGRESS":
            response["progress"] = task.info.get("progress", 0)
            response["step"] = task.info.get("step", "unknown")
        
        if task.status == "FAILURE":
            response["error"] = task.info.get("error", "Unknown error")
        
        return response
        
    except Exception as e:
        log_error(logger, e, {"task_id": task_id})
        raise HTTPException(status_code=500, detail="Failed to get task status")


@app.post("/ingest/batch", tags=["Ingestion"])
async def ingest_documents_batch(
    documents: List[Dict[str, Any]],
    org_id: str = "default"
):
    """Ingest multiple documents in batch"""
    try:
        # Queue batch processing task
        task = process_document_batch.delay(
            documents=documents,
            org_id=org_id
        )
        
        logger.info("Batch document processing task queued", 
                   task_id=task.id,
                   document_count=len(documents),
                   org_id=org_id)
        
        return {
            "task_id": task.id,
            "status": "queued",
            "message": f"Batch processing task queued for {len(documents)} documents",
            "document_count": len(documents),
            "org_id": org_id
        }
        
    except Exception as e:
        log_error(logger, e, {"document_count": len(documents), "org_id": org_id})
        raise HTTPException(status_code=500, detail="Failed to queue batch processing")


@app.post("/retrieve", tags=["Retrieval"])
async def retrieve_documents(
    query: str,
    top_k: int = 10,
    org_id: str = "default",
    use_bm25: bool = True,
    max_tokens: int = 4000,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve relevant documents without generating an answer"""
    start_time = time.time()
    try:
        # Initialize embeddings model
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Initialize retriever
        retriever = AdvancedRetriever(
            session=db,
            org_id=org_id,
            top_k=top_k,
            max_tokens=max_tokens,
            use_bm25_rerank=use_bm25
        )
        
        # Retrieve documents
        results = await retriever.retrieve(query, query_embedding, top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata,
                "source_title": result.source_title,
                "source_url": result.source_url
            })
        
        retrieval_time = time.time() - start_time
        
        # Record retrieval metrics
        record_retrieval_metrics(
            method="retrieve",
            status="success",
            duration=retrieval_time
        )
        
        logger.info("Document retrieval completed", 
                   query=query[:100],
                   org_id=org_id,
                   results_count=len(formatted_results),
                   top_score=max([r["score"] for r in formatted_results]) if formatted_results else 0)
        
        return {
            "query": query,
            "org_id": org_id,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "top_score": max([r["score"] for r in formatted_results]) if formatted_results else 0
        }
        
    except Exception as e:
        log_error(logger, e, {"query": query[:100], "org_id": org_id})
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.post("/query/intelligent", tags=["Intelligent QA"])
async def intelligent_query(
    query: str,
    top_k: int = 5,
    confidence_threshold: float = 0.7,
    max_attempts: int = 2,
    org_id: str = "default",
    db: AsyncSession = Depends(get_db)
):
    """Query documents using intelligent agent with query reformulation"""
    try:
        # Initialize intelligent agent
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        agent = IntelligentQAAgent(
            session=db,
            org_id=org_id,
            openai_api_key=settings.openai_api_key,
            confidence_threshold=confidence_threshold,
            max_attempts=max_attempts,
            model_name=settings.openai_model
        )
        
        # Get intelligent answer
        result = await agent.answer_question(
            query=query,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            max_attempts=max_attempts
        )
        
        # Record intelligent query metrics
        record_intelligent_query_metrics(
            method="intelligent_query",
            attempts=result["total_attempts"],
            reformulations=1 if result["reformulation_used"] else 0
        )
        
        # Record detailed metrics for the final result
        if result["attempts"]:
            final_attempt = max(result["attempts"], key=lambda x: x["confidence"])
            record_detailed_query_metrics(
                method="intelligent_query",
                tokens_in=final_attempt.get("total_tokens", 0),
                tokens_out=len(result["final_answer"].split()),  # Approximate output tokens
                retrieved_chunks=final_attempt.get("sources_count", 0),
                confidence=result["final_confidence"],
                processing_time=result["processing_time"]
            )
        
        logger.info("Intelligent query completed", 
                   query=query[:100],
                   final_confidence=result["final_confidence"],
                   total_attempts=result["total_attempts"],
                   reformulation_used=result["reformulation_used"])
        
        return {
            "query": query,
            "final_answer": result["final_answer"],
            "final_confidence": result["final_confidence"],
            "total_attempts": result["total_attempts"],
            "reformulation_used": result["reformulation_used"],
            "processing_time": result["processing_time"],
            "attempts": result["attempts"],
            "reformulation_reason": result.get("reformulation_reason"),
            "keywords_extracted": result.get("keywords_extracted"),
            "metadata_filters": result.get("metadata_filters")
        }
        
    except Exception as e:
        log_error(logger, e, {"query": query[:100], "org_id": org_id})
        raise HTTPException(status_code=500, detail="Failed to process intelligent query")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to KnowledgeOps AI",
        "version": settings.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
