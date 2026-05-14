import sys
sys.path.insert(0, '.')

from src.config import DATABASE_URL
import psycopg2

def init_db():
    """Create all tables"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Enable pgvector
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Policies table
        cursor.execute("""
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
        cursor.execute("""
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_cache (
                id SERIAL PRIMARY KEY,
                content_hash VARCHAR(255) UNIQUE,
                decision VARCHAR(50),
                cached_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_policies_embedding 
            ON policies USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_request_id 
            ON audit_logs(request_id);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database initialized successfully")
        return True
    
    except Exception as e:
        print(f"❌ Database init failed: {e}")
        return False

if __name__ == "__main__":
    init_db()
