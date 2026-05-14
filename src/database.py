import os
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from datetime import datetime
from src.config import DATABASE_URL, DEBUG

class ModeratorDB:
    """PostgreSQL + pgvector backend"""
    
    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Initialize database connection"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.cursor = self.conn.cursor()
            register_vector(self.conn)
            print("✓ Connected to PostgreSQL")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def init_schema(self):
        """Create tables on first run"""
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Policies table (vector embeddings)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS policies (
                id SERIAL PRIMARY KEY,
                policy_id VARCHAR(255) UNIQUE,
                title VARCHAR(500),
                description TEXT,
                embedding vector(384),
                category VARCHAR(100),
                severity INTEGER,
                examples TEXT[],
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Audit log table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                request_id VARCHAR(255),
                user_id VARCHAR(255),
                decision VARCHAR(50),
                confidence FLOAT,
                reasoning TEXT,
                policies_matched TEXT[],
                agent_trace JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Decision cache
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_cache (
                id SERIAL PRIMARY KEY,
                content_hash VARCHAR(255) UNIQUE,
                decision VARCHAR(50),
                cached_at TIMESTAMP DEFAULT NOW(),
                expires_at TIMESTAMP
            );
        """)
        
        # Index for vector search
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS policies_embedding_idx 
            ON policies USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        self.conn.commit()
        print("✓ Database schema initialized")
    
    def insert_policy(self, policy_id: str, title: str, description: str,
                      embedding: np.ndarray, category: str, severity: int,
                      examples: list):
        """Insert policy with embedding"""
        embedding_list = embedding.tolist()
        self.cursor.execute("""
            INSERT INTO policies (policy_id, title, description, embedding, category, severity, examples)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (policy_id) DO NOTHING;
        """, (policy_id, title, description, embedding_list, category, severity, examples))
        self.conn.commit()
    
    def search_policies(self, query_embedding: np.ndarray, limit: int = 5):
        """Semantic search over policies"""
        embedding_list = query_embedding.tolist()
        self.cursor.execute("""
            SELECT policy_id, title, description, category, severity, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM policies
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding_list, embedding_list, limit))
        return self.cursor.fetchall()
    
    def log_decision(self, request_id: str, user_id: str, decision: str,
                     confidence: float, reasoning: str, policies: list, trace: dict):
        """Write decision to audit log"""
        self.cursor.execute("""
            INSERT INTO audit_logs (request_id, user_id, decision, confidence, 
                                    reasoning, policies_matched, agent_trace)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """, (request_id, user_id, decision, confidence, reasoning, 
              policies, json.dumps(trace)))
        self.conn.commit()
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
