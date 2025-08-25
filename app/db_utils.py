"""
Database utility functions for KnowledgeOps AI
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload
import uuid
from datetime import datetime

from app.models_db import Document, Chunk, Conversation


async def create_document(
    session: AsyncSession,
    org_id: str,
    source: str,
    author: Optional[str] = None
) -> Document:
    """Create a new document"""
    document = Document(
        id=uuid.uuid4(),
        org_id=org_id,
        source=source,
        author=author,
        created_at=datetime.utcnow()
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def get_document_by_id(session: AsyncSession, doc_id: uuid.UUID) -> Optional[Document]:
    """Get document by ID with chunks"""
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.chunks))
        .where(Document.id == doc_id)
    )
    return result.scalar_one_or_none()


async def get_documents_by_org(
    session: AsyncSession, 
    org_id: str, 
    limit: int = 100,
    offset: int = 0
) -> List[Document]:
    """Get documents by organization ID"""
    result = await session.execute(
        select(Document)
        .where(Document.org_id == org_id)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def create_chunk(
    session: AsyncSession,
    doc_id: uuid.UUID,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None
) -> Chunk:
    """Create a new chunk with optional embedding"""
    chunk = Chunk(
        id=uuid.uuid4(),
        doc_id=doc_id,
        text=text,
        metadata=metadata,
        embedding=embedding
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)
    return chunk


async def get_chunks_by_document(
    session: AsyncSession,
    doc_id: uuid.UUID,
    limit: int = 1000
) -> List[Chunk]:
    """Get chunks by document ID"""
    result = await session.execute(
        select(Chunk)
        .where(Chunk.doc_id == doc_id)
        .order_by(Chunk.id)
        .limit(limit)
    )
    return result.scalars().all()


async def search_chunks_by_embedding(
    session: AsyncSession,
    query_embedding: List[float],
    org_id: Optional[str] = None,
    top_k: int = 10,
    similarity_threshold: float = 0.7
) -> List[Chunk]:
    """Search chunks by embedding similarity using pgvector"""
    # Build the query
    query = select(Chunk).join(Document).where(
        Chunk.embedding.isnot(None)
    )
    
    # Add organization filter if provided
    if org_id:
        query = query.where(Document.org_id == org_id)
    
    # Add similarity search using pgvector
    query = query.order_by(
        func.cosine_similarity(Chunk.embedding, query_embedding).desc()
    ).limit(top_k)
    
    result = await session.execute(query)
    chunks = result.scalars().all()
    
    # Filter by similarity threshold (if needed)
    # Note: This would require additional processing as cosine_similarity
    # is calculated in the database but we need to filter here
    return chunks


async def search_chunks_by_metadata(
    session: AsyncSession,
    org_id: str,
    metadata_filters: Dict[str, Any],
    limit: int = 100
) -> List[Chunk]:
    """Search chunks by metadata filters"""
    query = select(Chunk).join(Document).where(
        Document.org_id == org_id
    )
    
    # Add metadata filters
    for key, value in metadata_filters.items():
        query = query.where(Chunk.metadata.contains({key: value}))
    
    query = query.limit(limit)
    
    result = await session.execute(query)
    return result.scalars().all()


async def create_conversation(
    session: AsyncSession,
    org_id: str,
    user_id: str
) -> Conversation:
    """Create a new conversation"""
    conversation = Conversation(
        id=uuid.uuid4(),
        org_id=org_id,
        user_id=user_id,
        created_at=datetime.utcnow()
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def get_conversations_by_user(
    session: AsyncSession,
    org_id: str,
    user_id: str,
    limit: int = 50
) -> List[Conversation]:
    """Get conversations by user"""
    result = await session.execute(
        select(Conversation)
        .where(
            and_(
                Conversation.org_id == org_id,
                Conversation.user_id == user_id
            )
        )
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def get_document_stats(session: AsyncSession, org_id: str) -> Dict[str, int]:
    """Get document statistics for an organization"""
    # Count documents
    doc_count_result = await session.execute(
        select(func.count(Document.id))
        .where(Document.org_id == org_id)
    )
    doc_count = doc_count_result.scalar()
    
    # Count chunks
    chunk_count_result = await session.execute(
        select(func.count(Chunk.id))
        .join(Document)
        .where(Document.org_id == org_id)
    )
    chunk_count = chunk_count_result.scalar()
    
    # Count chunks with embeddings
    embedded_chunk_count_result = await session.execute(
        select(func.count(Chunk.id))
        .join(Document)
        .where(
            and_(
                Document.org_id == org_id,
                Chunk.embedding.isnot(None)
            )
        )
    )
    embedded_chunk_count = embedded_chunk_count_result.scalar()
    
    return {
        "total_documents": doc_count,
        "total_chunks": chunk_count,
        "embedded_chunks": embedded_chunk_count
    }


async def delete_document_and_chunks(
    session: AsyncSession,
    doc_id: uuid.UUID
) -> bool:
    """Delete a document and all its chunks"""
    try:
        # Get the document
        document = await get_document_by_id(session, doc_id)
        if not document:
            return False
        
        # Delete the document (chunks will be deleted due to cascade)
        await session.delete(document)
        await session.commit()
        return True
    except Exception:
        await session.rollback()
        return False
