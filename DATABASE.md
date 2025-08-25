# 🗄️ KnowledgeOps AI Database Setup

## Overview

This document describes the database schema and migration setup for KnowledgeOps AI, which uses PostgreSQL with the pgvector extension for vector similarity search.

## Database Schema

### Tables

#### 1. `documents` Table
Stores document metadata and source information.

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id VARCHAR(50) NOT NULL,
    source VARCHAR(255) NOT NULL,
    author VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

**Indexes:**
- `ix_documents_org_id` on `org_id`
- `idx_documents_org_id_created` on `(org_id, created_at)`

#### 2. `chunks` Table
Stores document chunks with embeddings for vector search.

```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1536)
);
```

**Indexes:**
- `ix_chunks_doc_id` on `doc_id`
- `idx_chunks_doc_id_metadata` on `(doc_id, metadata)`

#### 3. `conversations` Table
Stores conversation sessions for multi-tenant support.

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

**Indexes:**
- `ix_conversations_org_id` on `org_id`
- `ix_conversations_user_id` on `user_id`
- `idx_conversations_org_user` on `(org_id, user_id)`
- `idx_conversations_created` on `created_at`

## Prerequisites

### 1. PostgreSQL with pgvector
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-14-pgvector  # Adjust version as needed

# Or build from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

## Setup Instructions

### 1. Create Database
```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE knowledgeops;
CREATE USER knowledgeops WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE knowledgeops TO knowledgeops;
\q
```

### 2. Configure Environment
```bash
# Copy environment template
cp env.example .env

# Edit .env with your database settings
DATABASE_URL=postgresql+asyncpg://knowledgeops:password@localhost/knowledgeops
```

### 3. Run Migrations
```bash
# Test database setup
python test_db.py

# Run migrations
python migrate.py

# Or manually with Alembic
alembic upgrade head
```

## Migration Files

### Initial Migration (`0001_initial_schema.py`)
- Enables pgvector extension
- Creates all three tables with proper relationships
- Adds performance indexes
- Includes downgrade functionality

### Key Features
- **UUID Primary Keys**: For better distribution and security
- **Cascade Deletes**: Chunks are automatically deleted when documents are removed
- **JSONB Metadata**: Flexible metadata storage for chunks
- **Vector Embeddings**: 1536-dimensional vectors for similarity search
- **Multi-tenant Support**: Organization-based data isolation

## Database Utilities

### Core Functions

#### Document Management
```python
from app.db_utils import create_document, get_document_by_id, get_documents_by_org

# Create a document
doc = await create_document(session, org_id="org1", source="https://example.com/doc.pdf")

# Get document with chunks
doc = await get_document_by_id(session, doc_id)

# List documents by organization
docs = await get_documents_by_org(session, org_id="org1")
```

#### Chunk Management
```python
from app.db_utils import create_chunk, get_chunks_by_document

# Create a chunk with embedding
chunk = await create_chunk(
    session, 
    doc_id=doc.id, 
    text="Document content...",
    metadata={"page": 1, "section": "introduction"},
    embedding=[0.1, 0.2, ...]  # 1536-dimensional vector
)

# Get chunks for a document
chunks = await get_chunks_by_document(session, doc_id)
```

#### Vector Search
```python
from app.db_utils import search_chunks_by_embedding

# Search by embedding similarity
results = await search_chunks_by_embedding(
    session,
    query_embedding=[0.1, 0.2, ...],
    org_id="org1",
    top_k=10
)
```

#### Statistics
```python
from app.db_utils import get_document_stats

# Get organization statistics
stats = await get_document_stats(session, org_id="org1")
# Returns: {"total_documents": 10, "total_chunks": 150, "embedded_chunks": 120}
```

## API Endpoints

### Document Management
- `GET /documents?org_id=org1` - List documents
- `GET /documents/{document_id}` - Get document details
- `POST /ingest` - Ingest new document

### Statistics
- `GET /stats/{org_id}` - Get organization statistics

## Performance Considerations

### Indexes
- **Organization Indexes**: Fast filtering by `org_id`
- **Composite Indexes**: Efficient queries on `(org_id, created_at)`
- **Metadata Indexes**: JSONB queries on chunk metadata
- **Vector Indexes**: pgvector handles similarity search optimization

### Query Optimization
```sql
-- Efficient document listing with pagination
SELECT * FROM documents 
WHERE org_id = 'org1' 
ORDER BY created_at DESC 
LIMIT 100 OFFSET 0;

-- Vector similarity search with organization filter
SELECT c.*, d.org_id 
FROM chunks c 
JOIN documents d ON c.doc_id = d.id 
WHERE d.org_id = 'org1' 
  AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> '[0.1, 0.2, ...]'::vector 
LIMIT 10;
```

## Monitoring

### Database Metrics
- Document count by organization
- Chunk count and embedding coverage
- Query performance metrics
- Vector search latency

### Health Checks
```bash
# Check database connectivity
alembic current

# Check table sizes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename IN ('documents', 'chunks', 'conversations');
```

## Troubleshooting

### Common Issues

#### 1. pgvector Extension Not Available
```bash
# Check if pgvector is installed
psql -d knowledgeops -c "SELECT * FROM pg_available_extensions WHERE name = 'vector';"

# Install if missing
sudo apt install postgresql-14-pgvector
```

#### 2. Migration Failures
```bash
# Check migration status
alembic current

# Reset migrations (DANGER: loses data)
alembic downgrade base
alembic upgrade head
```

#### 3. Connection Issues
```bash
# Test database connection
psql -h localhost -U knowledgeops -d knowledgeops

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

### Performance Tuning

#### PostgreSQL Configuration
```sql
-- Increase shared buffers for better performance
ALTER SYSTEM SET shared_buffers = '256MB';

-- Optimize for vector operations
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';

-- Reload configuration
SELECT pg_reload_conf();
```

#### Vector Search Optimization
```sql
-- Create HNSW index for faster similarity search
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- Or use IVFFlat for better accuracy
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Backup and Recovery

### Backup Strategy
```bash
# Full database backup
pg_dump -h localhost -U knowledgeops -d knowledgeops > backup.sql

# Backup with custom format
pg_dump -h localhost -U knowledgeops -d knowledgeops -Fc > backup.dump

# Restore
psql -h localhost -U knowledgeops -d knowledgeops < backup.sql
pg_restore -h localhost -U knowledgeops -d knowledgeops backup.dump
```

### Automated Backups
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U knowledgeops -d knowledgeops > "backup_${DATE}.sql"
gzip "backup_${DATE}.sql"
```

## Security Considerations

### Database Security
- Use strong passwords for database users
- Enable SSL connections
- Restrict network access
- Regular security updates

### Data Privacy
- Organization-based data isolation
- Audit logging for data access
- Encryption at rest and in transit
- Regular data retention policies

## Next Steps

1. **Implement Document Processing**: Add text extraction and chunking logic
2. **Add Embedding Generation**: Integrate with embedding models
3. **Optimize Vector Search**: Add HNSW or IVFFlat indexes
4. **Add Authentication**: Implement user authentication and authorization
5. **Monitoring**: Add detailed performance monitoring and alerting
