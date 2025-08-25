-- Initialize KnowledgeOps AI database
-- This script runs when the PostgreSQL container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if it doesn't exist
-- (This is handled by Docker environment variables)

-- Create tables for the application
-- Note: These will be created by SQLAlchemy models, but we can add any custom setup here

-- Create indexes for better performance
-- (These will be created by SQLAlchemy as well)

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE knowledgeops TO knowledgeops;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO knowledgeops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO knowledgeops;
