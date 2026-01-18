-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create initial tables (will be managed by SQLAlchemy migrations in production)
-- This ensures pgvector is ready

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE rageval TO rageval;
