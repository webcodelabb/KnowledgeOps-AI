"""
SQLAlchemy database models for KnowledgeOps AI
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
import uuid
import sqlalchemy as sa

from app.database import Base


class Document(Base):
    """Document model for storing document metadata"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(String(50), nullable=False, index=True)
    source = Column(String(255), nullable=False)  # URL or file path
    author = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, org_id={self.org_id}, source={self.source})>"


class Chunk(Base):
    """Chunk model for storing document chunks with embeddings"""
    __tablename__ = "chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    metadata = Column(JSONB, nullable=True)  # JSON metadata for the chunk
    embedding = Column("embedding", sa.dialects.postgresql.VECTOR(1536), nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.doc_id})>"


class Conversation(Base):
    """Conversation model for storing chat sessions"""
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, org_id={self.org_id}, user_id={self.user_id})>"


# Create indexes for better performance
Index("idx_documents_org_id_created", Document.org_id, Document.created_at)
Index("idx_chunks_doc_id_metadata", Chunk.doc_id, Chunk.metadata)
Index("idx_conversations_org_user", Conversation.org_id, Conversation.user_id)
Index("idx_conversations_created", Conversation.created_at)
