"""Initial schema with pgvector and core tables

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create documents table
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', sa.String(length=50), nullable=False),
        sa.Column('source', sa.String(length=255), nullable=False),
        sa.Column('author', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create chunks table
    op.create_table('chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('doc_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('embedding', postgresql.VECTOR(1536), nullable=True),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create conversations table
    op.create_table('conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better performance
    op.create_index('ix_documents_org_id', 'documents', ['org_id'], unique=False)
    op.create_index('idx_documents_org_id_created', 'documents', ['org_id', 'created_at'], unique=False)
    
    op.create_index('ix_chunks_doc_id', 'chunks', ['doc_id'], unique=False)
    op.create_index('idx_chunks_doc_id_metadata', 'chunks', ['doc_id', 'metadata'], unique=False)
    
    op.create_index('ix_conversations_org_id', 'conversations', ['org_id'], unique=False)
    op.create_index('ix_conversations_user_id', 'conversations', ['user_id'], unique=False)
    op.create_index('idx_conversations_org_user', 'conversations', ['org_id', 'user_id'], unique=False)
    op.create_index('idx_conversations_created', 'conversations', ['created_at'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_conversations_created', table_name='conversations')
    op.drop_index('idx_conversations_org_user', table_name='conversations')
    op.drop_index('ix_conversations_user_id', table_name='conversations')
    op.drop_index('ix_conversations_org_id', table_name='conversations')
    
    op.drop_index('idx_chunks_doc_id_metadata', table_name='chunks')
    op.drop_index('ix_chunks_doc_id', table_name='chunks')
    
    op.drop_index('idx_documents_org_id_created', table_name='documents')
    op.drop_index('ix_documents_org_id', table_name='documents')
    
    # Drop tables
    op.drop_table('conversations')
    op.drop_table('chunks')
    op.drop_table('documents')
    
    # Drop pgvector extension
    op.execute('DROP EXTENSION IF EXISTS vector')
