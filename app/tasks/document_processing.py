"""
Document processing Celery tasks for KnowledgeOps AI
"""
import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import requests
from pathlib import Path

from celery import current_task
from langchain.document_loaders import (
    PyPDFLoader, 
    UnstructuredURLLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from sqlalchemy.ext.asyncio import AsyncSession

from app.celery_app import celery_app
from app.database import AsyncSessionLocal
from app.db_utils import create_document, create_chunk, get_document_by_id
from app.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


def download_file_from_url(url: str, temp_dir: str) -> str:
    """Download file from URL to temporary directory"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Determine file extension from URL or content-type
        content_type = response.headers.get('content-type', '')
        if 'pdf' in content_type.lower():
            ext = '.pdf'
        elif 'html' in content_type.lower():
            ext = '.html'
        else:
            ext = Path(urlparse(url).path).suffix or '.txt'
        
        # Create temporary file
        temp_file = os.path.join(temp_dir, f"download_{uuid.uuid4()}{ext}")
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("File downloaded successfully", url=url, temp_file=temp_file)
        return temp_file
        
    except Exception as e:
        logger.error("Failed to download file", url=url, error=str(e))
        raise


def get_document_loader(file_path: str, file_url: Optional[str] = None):
    """Get appropriate LangChain document loader based on file type"""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        return PyPDFLoader(file_path)
    elif file_ext in ['.html', '.htm']:
        return UnstructuredFileLoader(file_path)
    elif file_ext == '.txt':
        return TextLoader(file_path, encoding='utf-8')
    else:
        # Try unstructured loader for other file types
        return UnstructuredFileLoader(file_path)


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into chunks using LangChain text splitter"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        logger.info("Documents chunked successfully", 
                   original_count=len(documents), 
                   chunk_count=len(chunks),
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap)
        
        return chunks
        
    except Exception as e:
        logger.error("Failed to chunk documents", error=str(e))
        raise


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI embeddings"""
    try:
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress
            progress = min((i + batch_size) / len(texts) * 100, 100)
            current_task.update_state(
                state='PROGRESS',
                meta={'current': i + len(batch), 'total': len(texts), 'progress': progress}
            )
        
        logger.info("Embeddings generated successfully", 
                   text_count=len(texts), 
                   embedding_count=len(all_embeddings))
        
        return all_embeddings
        
    except Exception as e:
        logger.error("Failed to generate embeddings", error=str(e))
        raise


@celery_app.task(bind=True, name="app.tasks.document_processing.process_document")
def process_document(
    self,
    document_id: str,
    file_url: Optional[str] = None,
    file_path: Optional[str] = None,
    org_id: str = "default",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a document: load, chunk, embed, and store in database
    
    Args:
        document_id: UUID of the document record
        file_url: URL to download document from
        file_path: Local file path (if file_url not provided)
        org_id: Organization ID for multi-tenancy
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata for the document
    """
    task_id = self.request.id
    logger.info("Starting document processing task", 
               task_id=task_id, 
               document_id=document_id,
               file_url=file_url,
               file_path=file_path,
               org_id=org_id)
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'step': 'initializing', 'progress': 0})
        
        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Download or prepare file
            if file_url:
                logger.info("Downloading file from URL", url=file_url)
                self.update_state(state='PROGRESS', meta={'step': 'downloading', 'progress': 10})
                file_path = download_file_from_url(file_url, temp_dir)
            elif not file_path:
                raise ValueError("Either file_url or file_path must be provided")
            
            # Step 2: Load document using LangChain
            logger.info("Loading document with LangChain", file_path=file_path)
            self.update_state(state='PROGRESS', meta={'step': 'loading', 'progress': 20})
            
            loader = get_document_loader(file_path, file_url)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            logger.info("Document loaded successfully", 
                       document_count=len(documents),
                       total_chars=sum(len(doc.page_content) for doc in documents))
            
            # Step 3: Chunk documents
            logger.info("Chunking documents", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.update_state(state='PROGRESS', meta={'step': 'chunking', 'progress': 40})
            
            chunks = chunk_documents(documents, chunk_size, chunk_overlap)
            
            # Step 4: Generate embeddings
            logger.info("Generating embeddings", chunk_count=len(chunks))
            self.update_state(state='PROGRESS', meta={'step': 'embedding', 'progress': 60})
            
            texts = [chunk.page_content for chunk in chunks]
            embeddings = generate_embeddings(texts)
            
            # Step 5: Store in database
            logger.info("Storing chunks in database", chunk_count=len(chunks))
            self.update_state(state='PROGRESS', meta={'step': 'storing', 'progress': 80})
            
            # Use async context manager for database operations
            async def store_chunks():
                async with AsyncSessionLocal() as session:
                    # Get the document record
                    doc_uuid = uuid.UUID(document_id)
                    document = await get_document_by_id(session, doc_uuid)
                    
                    if not document:
                        raise ValueError(f"Document {document_id} not found")
                    
                    # Store chunks with embeddings
                    stored_chunks = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        # Prepare chunk metadata
                        chunk_metadata = {
                            "chunk_index": i,
                            "chunk_size": len(chunk.page_content),
                            "source": chunk.metadata.get("source", file_url or file_path),
                            "page": chunk.metadata.get("page"),
                            **(metadata or {})
                        }
                        
                        # Create chunk record
                        chunk_record = await create_chunk(
                            session=session,
                            doc_id=doc_uuid,
                            text=chunk.page_content,
                            metadata=chunk_metadata,
                            embedding=embedding
                        )
                        
                        stored_chunks.append(chunk_record.id)
                        
                        # Update progress
                        progress = 80 + (i / len(chunks)) * 20
                        self.update_state(
                            state='PROGRESS',
                            meta={'step': 'storing', 'progress': progress, 'current_chunk': i + 1}
                        )
                    
                    logger.info("Chunks stored successfully", 
                               document_id=document_id,
                               chunk_count=len(stored_chunks))
                    
                    return stored_chunks
            
            # Run the async function
            import asyncio
            stored_chunk_ids = asyncio.run(store_chunks())
            
            # Task completed successfully
            result = {
                "status": "completed",
                "document_id": document_id,
                "chunk_count": len(stored_chunk_ids),
                "chunk_ids": [str(chunk_id) for chunk_id in stored_chunk_ids],
                "file_url": file_url,
                "file_path": file_path,
                "org_id": org_id,
                "metadata": metadata
            }
            
            logger.info("Document processing completed successfully", 
                       task_id=task_id,
                       document_id=document_id,
                       chunk_count=len(stored_chunk_ids))
            
            self.update_state(state='SUCCESS', meta={'step': 'completed', 'progress': 100})
            return result
            
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        logger.error(error_msg, 
                    task_id=task_id,
                    document_id=document_id,
                    error=str(e),
                    exc_info=True)
        
        # Update task state with error
        self.update_state(
            state='FAILURE',
            meta={
                'error': error_msg,
                'document_id': document_id,
                'file_url': file_url,
                'file_path': file_path
            }
        )
        
        # Re-raise the exception
        raise


@celery_app.task(bind=True, name="app.tasks.document_processing.process_document_batch")
def process_document_batch(
    self,
    documents: List[Dict[str, Any]],
    org_id: str = "default"
) -> Dict[str, Any]:
    """
    Process multiple documents in batch
    
    Args:
        documents: List of document specifications
        org_id: Organization ID for multi-tenancy
    """
    task_id = self.request.id
    logger.info("Starting batch document processing", 
               task_id=task_id,
               document_count=len(documents),
               org_id=org_id)
    
    try:
        results = []
        failed_documents = []
        
        for i, doc_spec in enumerate(documents):
            try:
                # Create document record first
                async def create_doc_record():
                    async with AsyncSessionLocal() as session:
                        return await create_document(
                            session=session,
                            org_id=org_id,
                            source=doc_spec.get('file_url') or doc_spec.get('file_path', ''),
                            author=doc_spec.get('metadata', {}).get('author')
                        )
                
                import asyncio
                document = asyncio.run(create_doc_record())
                
                # Process the document
                result = process_document.delay(
                    document_id=str(document.id),
                    file_url=doc_spec.get('file_url'),
                    file_path=doc_spec.get('file_path'),
                    org_id=org_id,
                    chunk_size=doc_spec.get('chunk_size', 800),
                    chunk_overlap=doc_spec.get('chunk_overlap', 100),
                    metadata=doc_spec.get('metadata')
                )
                
                results.append({
                    "document_id": str(document.id),
                    "task_id": result.id,
                    "status": "queued"
                })
                
                logger.info("Document queued for processing", 
                           document_id=str(document.id),
                           task_id=result.id)
                
            except Exception as e:
                logger.error("Failed to queue document", 
                           document_spec=doc_spec,
                           error=str(e))
                failed_documents.append({
                    "document_spec": doc_spec,
                    "error": str(e)
                })
            
            # Update progress
            progress = (i + 1) / len(documents) * 100
            self.update_state(
                state='PROGRESS',
                meta={'current': i + 1, 'total': len(documents), 'progress': progress}
            )
        
        batch_result = {
            "status": "completed",
            "total_documents": len(documents),
            "queued_documents": len(results),
            "failed_documents": len(failed_documents),
            "results": results,
            "failed_documents": failed_documents
        }
        
        logger.info("Batch document processing completed", 
                   task_id=task_id,
                   queued_count=len(results),
                   failed_count=len(failed_documents))
        
        return batch_result
        
    except Exception as e:
        error_msg = f"Batch document processing failed: {str(e)}"
        logger.error(error_msg, task_id=task_id, error=str(e), exc_info=True)
        
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg, 'documents': documents}
        )
        
        raise
